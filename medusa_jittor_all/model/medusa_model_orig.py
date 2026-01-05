import os
import json
import time
import warnings

import jittor as jt
from jittor import nn

from transformers import PretrainedConfig, AutoTokenizer, AutoConfig

from .modeling_llama_jt import LlamaForCausalLM as KVLlamaForCausalLM
from .medusa_mask import pad_medusa_mask
from .utils_jt import (
    generate_medusa_buffers,
    reset_medusa_mode,
    initialize_medusa,
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    prepare_medusa_attention_mask,
)
from .kv_cache import initialize_past_key_values, reset_current_length_data
from .medusa_choices import *


def _use_cuda() -> bool:
    try:
        return bool(getattr(jt.flags, "use_cuda", 0))
    except Exception:
        return False


def _to_cuda(x: jt.Var) -> jt.Var:
    if x is None:
        return None
    if not isinstance(x, jt.Var):
        return x
    if not _use_cuda():
        return x
    try:
        # some builds expose .is_cuda as bool / callable
        ic = getattr(x, "is_cuda", None)
        if callable(ic):
            if ic():
                return x
        elif isinstance(ic, bool) and ic:
            return x
    except Exception:
        pass
    try:
        return x.cuda()
    except Exception:
        return x


class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class FastLinear(nn.Linear):
    """
    高性能 Linear：
    - 不强制 fp32（保留 fp16/bf16 TensorCore 路径）
    - 用 matmul 避免某些 jittor Linear 的 broadcast_to*multiply*reduce.add 低效路径
    """
    def execute(self, x):
        w = self.weight  # [out, in]
        b = self.bias
        if hasattr(x, "dtype") and hasattr(w, "dtype") and x.dtype != w.dtype:
            x = x.astype(w.dtype)

        if len(x.shape) == 3:
            B, S, H = x.shape
            y2 = jt.matmul(x.reshape((-1, H)), w.transpose())  # [B*S, out]
            y = y2.reshape((B, S, y2.shape[-1]))
        else:
            y = jt.matmul(x, w.transpose())

        if b is not None:
            if hasattr(y, "dtype") and hasattr(b, "dtype") and y.dtype != b.dtype:
                y = y + b.astype(y.dtype)
            else:
                y = y + b
        return y


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = FastLinear(hidden_size, hidden_size)
        # 初始化成近似 identity（训练 head-only 时更稳定）
        self.linear.weight.zero_()
        if self.linear.bias is not None:
            self.linear.bias.zero_()
        self.act = nn.SiLU()

    def execute(self, x):
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.medusa = int(getattr(config, "medusa_num_heads", 5))
        self.medusa_num_layers = int(getattr(config, "medusa_num_layers", 1))

        self.base_model_name_or_path = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", "")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, local_files_only=True)

        def _make_head():
            layers = [ResBlock(self.hidden_size) for _ in range(self.medusa_num_layers)]
            layers.append(FastLinear(self.hidden_size, self.vocab_size, bias=False))
            return nn.Sequential(*layers)

        self.medusa_head = nn.ModuleList([_make_head() for _ in range(self.medusa)])

        # ---- perf caches ----
        self._cached_medusa_buffers = {}  # key: tuple(tuple(path),...) -> buffers (on cuda)
        self._kv_cache_pack = None        # (cache_len, dtype, past_key_values, pkv_data, cur_len_data)

    @property
    def base_model(self):
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=True)

        if "medusa_num_heads" in kwargs:
            config.medusa_num_heads = kwargs.pop("medusa_num_heads")
        elif not hasattr(config, "medusa_num_heads"):
            config.medusa_num_heads = 5

        if "medusa_num_layers" in kwargs:
            config.medusa_num_layers = kwargs.pop("medusa_num_layers")
        elif not hasattr(config, "medusa_num_layers"):
            config.medusa_num_layers = 1

        model = cls(config)

        if os.path.exists(pretrained_model_name_or_path):
            model.load_base_model(pretrained_model_name_or_path)

        return model

    def get_tokenizer(self):
        return self.tokenizer

    def load_base_model(self, path: str):
        import importlib
        try:
            import torch
            if not hasattr(torch, "version"):
                torch.version = importlib.import_module("torch.version")
            if not hasattr(torch, "_utils"):
                torch._utils = importlib.import_module("torch._utils")
        except ImportError:
            print("Warning: Torch is required for loading weights but was not found.")
            return

        print(f"Loading base model weights from {path}...")

        shard_files = []
        if os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
            with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            shard_files = [os.path.join(path, f) for f in shard_files]
        elif os.path.exists(os.path.join(path, "pytorch_model.bin")):
            shard_files = [os.path.join(path, "pytorch_model.bin")]
        elif os.path.exists(os.path.join(path, "model.safetensors.index.json")):
            with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            shard_files = [os.path.join(path, f) for f in shard_files]
        elif os.path.exists(os.path.join(path, "model.safetensors")):
            shard_files = [os.path.join(path, "model.safetensors")]
        else:
            print(f"Warning: No weights found in {path}")
            return

        for shard_file in shard_files:
            print(f"Loading shard: {shard_file}")
            if shard_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(shard_file)
            else:
                state_dict = torch.load(shard_file, map_location="cpu")

            jt_state_dict = {}
            for k, v in state_dict.items():
                jt_state_dict[k] = _to_cuda(jt.array(v.to(torch.float16).cpu().numpy()))

            self.load_state_dict(jt_state_dict)
            del state_dict
            del jt_state_dict
            jt.gc()

        print("✅ Base model weights loaded.")

    def load_medusa_head(self, path: str):
        if not os.path.exists(path):
            print(f"Medusa head path {path} not found!")
            return

        print(f"Loading Medusa head from {path}")
        state_dict = jt.load(path)

        medusa_dict = {k: v for k, v in state_dict.items() if "medusa_head" in k}
        if not medusa_dict:
            print("Loading directly into medusa_head module...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if v.dtype == jt.float32:
                    v = v.astype(jt.float16)
                if k.startswith("heads."):
                    new_state_dict[k[6:]] = _to_cuda(v)
                else:
                    new_state_dict[k] = _to_cuda(v)
            self.medusa_head.load_state_dict(new_state_dict)
        else:
            for k, v in medusa_dict.items():
                if v.dtype == jt.float32:
                    medusa_dict[k] = _to_cuda(v.astype(jt.float16))
                else:
                    medusa_dict[k] = _to_cuda(v)
            self.load_state_dict(medusa_dict)

    def execute(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        if not medusa_forward:
            # ✅ 普通 decode / prefill：关闭 medusa mask
            try:
                self.model.medusa_mask = None
            except Exception:
                pass
            return super().execute(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )

        # ✅ critical: base + head 全部 no_grad；不要 clone hidden_states
        with jt.no_grad():
            outputs = self.model(
                input_ids=_to_cuda(input_ids),
                attention_mask=_to_cuda(attention_mask),
                past_key_values=past_key_values,
                position_ids=_to_cuda(position_ids),
                **kwargs,
            )
            hidden_states = outputs[0]
            if output_orig:
                orig = self.lm_head(hidden_states)

            medusa_logits = [self.medusa_head[i](hidden_states) for i in range(self.medusa)]
            stacked = jt.stack(medusa_logits, dim=0)

            if output_orig:
                return stacked, outputs, orig
            return stacked

    def get_medusa_choice(self, model_name: str):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def _get_cached_buffers(self, medusa_choices):
        key = tuple(tuple(x) for x in medusa_choices)
        buf = self._cached_medusa_buffers.get(key, None)
        if buf is None:
            buf = generate_medusa_buffers(medusa_choices, device="cuda" if _use_cuda() else "cpu")
            self._cached_medusa_buffers[key] = buf
        return buf

    def _get_or_create_kv(self, cache_len: int, dtype=jt.float16):
        pack = self._kv_cache_pack
        if pack is not None:
            old_len, old_dtype, past_key_values, pkv_data, cur_len = pack
            if int(old_len) == int(cache_len) and old_dtype == dtype:
                reset_current_length_data(cur_len)  # zero without realloc
                return past_key_values, pkv_data, cur_len

        # allocate new
        past_key_values, pkv_data, cur_len = initialize_past_key_values(
            self.base_model,
            max_cache_len=int(cache_len),
            dtype=dtype,
        )
        self._kv_cache_pack = (int(cache_len), dtype, past_key_values, pkv_data, cur_len)
        return past_key_values, pkv_data, cur_len

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        medusa_choices=None,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
        top_p=0.8,
        sampling='typical',
        fast=True,
        tree_step: int = 64,
        tree_pad_steps: int = 4,
        safe_pad: int = 32,
        max_cache_len: int = None,
        eos_check_interval: int = 32,  # ✅ reduce sync
        debug: bool = False,
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"

        # ✅ ensure CUDA for input
        input_ids = _to_cuda(input_ids.cast(jt.int32))
        attention_mask = _to_cuda(attention_mask) if attention_mask is not None else None

        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)
            medusa_choices = [c for c in medusa_choices if len(c) <= int(self.medusa)]

        medusa_buffers = self._get_cached_buffers(medusa_choices)

        tree_step = int(tree_step)
        tree_pad_steps = int(tree_pad_steps)
        safe_pad = int(safe_pad)
        tree_pad = tree_pad_steps * tree_step + safe_pad
        kv_headroom = tree_step + 16

        prefill_len = int(input_ids.shape[1])
        max_steps = int(max_steps)

        # “有效上下文”仍然按模型配置控制（prompt 截断）
        ctx_cap = int(getattr(self.base_model.config, "max_position_embeddings", 2048) or 2048)

        # ✅ KV cache 长度不要硬对齐 ctx_cap（Medusa tree decoding 需要额外 headroom）
        # 若不给 max_cache_len：根据 (prefill + max_steps + tree_pad + headroom) 估一个合理值，并向上取 256 对齐
        if max_cache_len is None:
            need = prefill_len + max_steps + tree_pad + kv_headroom + 16
            cache_len = max(ctx_cap, int(need))
            # align up to 256
            cache_len = ((cache_len + 255) // 256) * 256
            # 常见安全上限：4096（你也可显式传 --kv_cache_len 4096）
            cache_len = max(cache_len, 4096 if ctx_cap <= 2048 else cache_len)
        else:
            cache_len = int(max_cache_len)

        # ✅ ensure rope cache can cover cache_len
        try:
            if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "ensure_rope_cache"):
                self.base_model.model.ensure_rope_cache(int(cache_len))
        except Exception:
            pass

        # prompt 截断（只按 ctx_cap；不需要扣 tree_pad —— 因为 KV cache 已经更长）
        if prefill_len > ctx_cap:
            input_ids = input_ids[:, -ctx_cap:]
            prefill_len = int(input_ids.shape[1])

        # max_steps 也不要被 ctx_cap 误伤（只要不超过 KV cache_len 即可）
        max_steps_budget = int(cache_len) - prefill_len - tree_pad - kv_headroom
        if max_steps_budget < 1:
            max_steps_budget = 1
        max_steps = min(max_steps, max_steps_budget)

        past_key_values, past_key_values_data, current_length_data = self._get_or_create_kv(
            cache_len=int(cache_len),
            dtype=jt.float16,
        )

        # ---- initialize (prefill) ----
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        # ✅ Optimization: Do NOT set self.model.medusa_mask.
        # Instead, we construct the full mask once per step and pass it as attention_mask.
        # This avoids reconstructing it 32 times (once per layer) in attention_fixed.py.
        self.model.medusa_mask = None 

        # Get capacity from KV cache
        max_kv_len_capacity = int(past_key_values[0][0].shape[2])
        medusa_mask_small = medusa_buffers["medusa_attn_mask"]
        q_len_tree = int(medusa_mask_small.shape[-1])

        new_token = 0
        eos_id = self.tokenizer.eos_token_id
        
        # ✅ Track current length as python int for fast path
        cur_len_int = prefill_len

        for idx in range(max_steps):
            if debug:
                jt.sync_all(True)
                t0 = time.time()

            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_threshold=posterior_threshold,
                posterior_alpha=posterior_alpha,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            if debug:
                jt.sync_all(True)
                t1 = time.time()

            # ✅ Pre-compute mask once with Bucketing
            # Bucket size 256 to reduce shape variations while keeping FLOPs low
            bucket_size = 256
            needed_len = cur_len_int + q_len_tree
            bucket_len = ((needed_len + bucket_size - 1) // bucket_size) * bucket_size
            # Clamp to max capacity
            bucket_len = min(bucket_len, max_kv_len_capacity)

            full_attn_mask = prepare_medusa_attention_mask(
                medusa_mask_small, 
                cache_len=cur_len_int, 
                max_kv_len=bucket_len, 
                q_len=q_len_tree
            )

            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
                current_len_cpu=cur_len_int,  # ✅ Pass int length
                attention_mask=full_attn_mask, # ✅ Pass pre-computed mask (bucketed)
            )

            if debug:
                jt.sync_all(True)
                t2 = time.time()

            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p, sampling, fast
            )
            
            # ✅ Update int length
            cur_len_int += (accept_length + 1)

            if debug:
                jt.sync_all(True)
                t3 = time.time()

            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            if debug:
                jt.sync_all(True)
                t4 = time.time()
                print(f"[Step {idx}] GenCand: {t1-t0:.4f}s | TreeFwd: {t2-t1:.4f}s | EvalPost: {t3-t2:.4f}s | UpdateKV: {t4-t3:.4f}s | AcceptLen: {accept_length}")

            # ✅ EOS check: low-frequency sync to avoid killing throughput
            if eos_id is not None and eos_check_interval > 0 and (idx % eos_check_interval == 0):
                tail = input_ids[0, -(accept_length + 1):].cast(jt.int32)
                hit = (tail == int(eos_id)).any()
                try:
                    if bool(hit.item()):
                        break
                except Exception:
                    pass

        reset_medusa_mode(self)
        # ✅ 清掉，避免影响后续普通 decode
        try:
            self.model.medusa_mask = None
        except Exception:
            pass
        return input_ids


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        KVLlamaForCausalLM.__init__(self, config)
        MedusaModelABC.__init__(self, config, **kwargs)


class MedusaModel:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            try:
                # Try loading config from the path directly
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=True)
            except Exception:
                try:
                    # Fallback: try MedusaConfig
                    config = MedusaConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=True)
                except Exception:
                    # Fallback 2: if path is a checkpoint folder without config, try loading from base_model
                    # This happens when loading a Medusa checkpoint that only has weights but relies on base model config
                    print(f"[MedusaModel] Warning: config.json not found in {pretrained_model_name_or_path}. Trying to load config from base model path if available in kwargs or config.json inside checkpoint.")
                    
                    # Check if there is a config.json inside the folder anyway (sometimes AutoConfig fails if it's not a standard HF repo)
                    if os.path.exists(os.path.join(pretrained_model_name_or_path, "config.json")):
                         config = AutoConfig.from_pretrained(os.path.join(pretrained_model_name_or_path, "config.json"), local_files_only=True)
                    else:
                        # If we can't find config, we might need the user to provide base_model_name_or_path
                        # But often Medusa checkpoints are just weights.
                        # Let's assume the user provided 'base_model_name_or_path' in kwargs or we can infer it?
                        # Actually, for MedusaModel.from_pretrained, we usually expect a full model or a medusa adapter.
                        
                        # If this is a Medusa checkpoint (adapter), we should load the base model config.
                        # Let's try to read adapter_config.json if it exists (PEFT style) or just fail gracefully.
                        pass
                        
                    if config is None:
                         # Last resort: if we are loading a split checkpoint, maybe the config is in the parent or we should use the base model config provided in arguments?
                         # In bench_mtbench_all.py, we pass --baseline_model.
                         # But here we don't have access to that arg directly unless passed in kwargs.
                         pass
                         
        # If config is still None, re-raise the last exception or a new one
        if config is None:
             # Retry original logic to raise the proper error
             try:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=True)
             except Exception:
                config = MedusaConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=True)

        return MedusaModelLlama.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)