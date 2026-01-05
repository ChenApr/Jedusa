import argparse, json, time, os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shortuuid
import torch
import numpy as np  # ✅ 必须加
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from fastchat.conversation import get_conv_template

# ✅ 确保能 import medusa_jittor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medusa_jittor.model.medusa_model import MedusaModel


def _to_torch_on_device(x, device):
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, device=device)

def load_medusa_head_weights_strict(model, weights_path: str):
    # weights_path: .pkl / .npz / .pt / .bin
    state_dict = None

    if weights_path.endswith(".npz"):
        w = np.load(weights_path)
        state_dict = {k: torch.from_numpy(v) for k, v in w.items()}

    elif weights_path.endswith((".pkl", ".pt", ".bin")):
        if weights_path.endswith(".pkl"):
            import jittor as jt
            weights = jt.load(weights_path)
            state_dict = {k: (torch.from_numpy(v.numpy()) if hasattr(v, "numpy") else torch.from_numpy(v))
                          for k, v in weights.items()}
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported medusa weight format: {weights_path}")

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("heads."):
            k = k[len("heads."):]
        if k.startswith("medusa_head."):
            k = k[len("medusa_head."):]
        cleaned[k] = v

    cleaned = {k: _to_torch_on_device(v, model.device) for k, v in cleaned.items()}
    model.medusa_head.load_state_dict(cleaned, strict=True)
    print("✅ Medusa head weights loaded (strict=True)")

# ----------------------------
# IO
# ----------------------------
def load_questions(path: str, limit: Optional[int] = None, start: int = 0) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    items = items[start:]
    if limit is not None:
        items = items[:limit]
    return items


def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------
# Timing helpers
# ----------------------------
def cuda_time(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return out, (t1 - t0)


# ----------------------------
# Stop criteria
# ----------------------------
class StopOnSubstrings(StoppingCriteria):
    """
    Lightweight substring stopping. Each step decodes only last `window` tokens of newly generated part.
    """
    def __init__(self, tokenizer, stop_strs: List[str], start_len: int, window: int = 64):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strs = [s for s in stop_strs if s]
        self.start_len = start_len
        self.window = window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_strs:
            return False
        # Only examine last window tokens in generated part
        gen = input_ids[0, self.start_len:]
        if gen.numel() <= 0:
            return False
        tail = gen[-self.window:].tolist()
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return any(s in text for s in self.stop_strs)


def get_conv_stop(conv) -> Tuple[List[str], List[int]]:
    """
    Extract stop strings / stop token ids from FastChat template (best-effort).
    Different versions may have slightly different fields.
    """
    stop_strs = []
    stop_token_ids = []

    if hasattr(conv, "stop_str") and conv.stop_str:
        if isinstance(conv.stop_str, str):
            stop_strs.append(conv.stop_str)
        else:
            stop_strs.extend(list(conv.stop_str))

    if hasattr(conv, "stop_token_ids") and conv.stop_token_ids:
        stop_token_ids.extend(list(conv.stop_token_ids))

    # Some templates rely on sep / sep2 as implicit stops
    # (we only add them if they look like real separators)
    for name in ["sep", "sep2"]:
        if hasattr(conv, name):
            s = getattr(conv, name)
            if isinstance(s, str) and len(s) > 0 and s not in stop_strs:
                # Avoid overly common whitespace-like separators
                if s.strip() != "":
                    stop_strs.append(s)

    return stop_strs, stop_token_ids


def cut_by_stop(text: str, stop_strs: List[str]) -> str:
    """
    Post-process: truncate at earliest occurrence of any stop string.
    """
    if not stop_strs:
        return text
    idxs = [text.find(s) for s in stop_strs if s and text.find(s) != -1]
    if not idxs:
        return text
    cut = min(idxs)
    return text[:cut]


# ----------------------------
# Prompt building
# ----------------------------
def build_prompt(conv_name: str, user_messages: List[str], assistant_messages: List[str]) -> Tuple[str, Any]:
    conv = get_conv_template(conv_name).copy()
    for i, umsg in enumerate(user_messages):
        conv.append_message(conv.roles[0], umsg)
        if i < len(assistant_messages):
            conv.append_message(conv.roles[1], assistant_messages[i])
        else:
            conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


# ----------------------------
# Token counting
# ----------------------------
def count_output_tokens(tok, text: str) -> int:
    #统一：用 encode(output_text) 计数，baseline & medusa 都一致
    return len(tok.encode(text, add_special_tokens=False))


# ----------------------------
# Input truncation
# ----------------------------
def infer_max_ctx(model, tokenizer, override: Optional[int]) -> int:
    if override is not None:
        return int(override)

    # Prefer model config
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
        return int(model.config.max_position_embeddings)

    # Fallback to tokenizer
    m = getattr(tokenizer, "model_max_length", None)
    if m is not None and isinstance(m, int) and m < 10**6:
        return int(m)

    # Safe fallback
    return 2048


def tokenize_prompt(
    tok,
    prompt: str,
    device: torch.device,
    max_input_len: int,
) -> Dict[str, torch.Tensor]:
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, int(max_input_len)),
    )
    return {k: v.to(device) for k, v in enc.items()}


# ----------------------------
# Generation: baseline
# ----------------------------
def gen_baseline(
    model,
    tok,
    prompt: str,
    conv_stop_strs: List[str],
    conv_stop_token_ids: List[int],
    max_ctx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, int]:
    # Leave room for output tokens (+ small safety)
    safety = 16
    max_in = max(1, max_ctx - max_new_tokens - safety)
    inputs = tokenize_prompt(tok, prompt, model.device, max_in)

    do_sample = temperature > 0
    eos_ids = []
    if model.config.eos_token_id is not None:
        eos_ids.append(int(model.config.eos_token_id))
    eos_ids.extend([int(x) for x in conv_stop_token_ids if x is not None])

    # stopping criteria for stop_str (best-effort)
    start_len = inputs["input_ids"].shape[1]
    stop_criteria = StoppingCriteriaList()
    if conv_stop_strs:
        stop_criteria.append(StopOnSubstrings(tok, conv_stop_strs, start_len=start_len, window=64))

    out_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,  # HF supports list in newer versions
        pad_token_id=model.config.eos_token_id,
        stopping_criteria=stop_criteria if len(stop_criteria) > 0 else None,
    )

    gen_ids = out_ids[0, start_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    text = cut_by_stop(text, conv_stop_strs).strip()

    n_out = count_output_tokens(tok, text)
    return text, n_out


# ----------------------------
# Generation: medusa
# ----------------------------
def gen_medusa(
    medusa_model: MedusaModel,
    tok,
    prompt: str,
    conv_stop_strs: List[str],
    max_ctx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    # The following are Medusa-implementation constraints (tree attention padding)
    tree_step: int = 64,
    tree_pad_steps: int = 4,
    safe_pad: int = 32,
) -> Tuple[str, int]:
    # Medusa tree attention needs extra reserved positions.
    tree_pad = tree_pad_steps * tree_step + safe_pad
    max_in = max(1, max_ctx - tree_pad - 16)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)
    input_ids = enc["input_ids"].to(medusa_model.base_model.device)
    in_len = input_ids.shape[1]

    # Also cap steps so we don't overflow ctx budget
    max_steps_allow = max(1, min(max_new_tokens, max_ctx - tree_pad - in_len))

    out = medusa_model.medusa_generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        max_steps=max_steps_allow,
    )

    # medusa_generate may return str or a generator yielding deltas / dicts
    if isinstance(out, str):
        out_text = out
    else:
        deltas = []
        last_text = ""
        for item in out:
            if isinstance(item, str):
                deltas.append(item)
            elif isinstance(item, dict):
                if "delta" in item and item["delta"]:
                    deltas.append(item["delta"])
                elif "text" in item and item["text"]:
                    last_text = item["text"]
                elif "content" in item and item["content"]:
                    last_text = item["content"]

        out_text = "".join(deltas) if len(deltas) > 0 else last_text

    out_text = cut_by_stop(out_text, conv_stop_strs).strip()
    n_out = count_output_tokens(tok, out_text)
    return out_text, n_out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=str, required=True)
    ap.add_argument("--mode", choices=["baseline", "medusa"], required=True)
    ap.add_argument("--model", type=str, required=True, help="baseline模型路径或medusa模型路径")
    ap.add_argument("--base-model", type=str, default=None, help="Tokenizer/base config path for Medusa-1 head-only")
    ap.add_argument("--conv", type=str, default="vicuna_v1.1", help="FastChat conv template name")

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)

    ap.add_argument("--max_ctx", type=int, default=None, help="Override model max context length")
    ap.add_argument("--out_json", type=str, default=None, help="Write metrics to json")

    # Medusa padding knobs (keep defaults unless needed)
    ap.add_argument("--tree_step", type=int, default=64)
    ap.add_argument("--tree_pad_steps", type=int, default=4)
    ap.add_argument("--safe_pad", type=int, default=32)
    ap.add_argument("--answer_file", type=str, default=None, help="Write FastChat model_answer jsonl")
    ap.add_argument("--model_id", type=str, default=None, help="model_id field for FastChat judge")
    ap.add_argument("--medusa_weights", type=str, default=None, help="Path to jittor-trained medusa head weights (.pkl)")
    ap.add_argument("--medusa_num_heads", type=int, default=3, help="Number of Medusa heads (must match training)")

    args = ap.parse_args()

    def append_answer(answer_file, question_id, model_id, turns):
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        ans_json = {
            "question_id": question_id,
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": [{"index": 0, "turns": turns}],
            "tstamp": time.time(),
        }
        with open(answer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

    qs = load_questions(args.questions, limit=args.limit, start=args.start)
    assert len(qs) > 0, "no questions loaded"

    torch.set_grad_enabled(False)

    # tokenizer path
    tok_path = args.model
    if args.mode == "medusa" and args.base_model is not None:
        tok_path = args.base_model
    tok = AutoTokenizer.from_pretrained(tok_path, use_fast=False)

    # # Load model
    # if args.mode == "baseline":
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #     )
    #     model.eval()
    #     max_ctx = infer_max_ctx(model, tok, args.max_ctx)
    #     gen_fn = lambda prompt, conv_stop_strs, conv_stop_token_ids: gen_baseline(
    #         model=model,
    #         tok=tok,
    #         prompt=prompt,
    #         conv_stop_strs=conv_stop_strs,
    #         conv_stop_token_ids=conv_stop_token_ids,
    #         max_ctx=max_ctx,
    #         max_new_tokens=args.max_new_tokens,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #     )
    # else:
    #     medusa = MedusaModel.from_pretrained(
    #         args.model,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #     )
    #     medusa.eval()
    #     # max_ctx should come from base model config
    #     max_ctx = infer_max_ctx(medusa.base_model, tok, args.max_ctx)
    #     gen_fn = lambda prompt, conv_stop_strs, _conv_stop_token_ids: gen_medusa(
    #         medusa_model=medusa,
    #         tok=tok,
    #         prompt=prompt,
    #         conv_stop_strs=conv_stop_strs,
    #         max_ctx=max_ctx,
    #         max_new_tokens=args.max_new_tokens,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #         tree_step=args.tree_step,
    #         tree_pad_steps=args.tree_pad_steps,
    #         safe_pad=args.safe_pad,
    #     )
    if args.mode == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        max_ctx = infer_max_ctx(model, tok, args.max_ctx)
        gen_fn = lambda prompt, conv_stop_strs, conv_stop_token_ids: gen_baseline(
            model=model,
            tok=tok,
            prompt=prompt,
            conv_stop_strs=conv_stop_strs,
            conv_stop_token_ids=conv_stop_token_ids,
            max_ctx=max_ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    else:
        assert args.medusa_weights is not None, "medusa mode requires --medusa_weights"
        medusa = MedusaModel.from_pretrained(
            args.model,                      # ✅ 仍然是 base vicuna 路径
            medusa_num_heads=args.medusa_num_heads,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        medusa.eval()

        # ✅ 严格加载你训练出的 head 权重
        load_medusa_head_weights_strict(medusa, args.medusa_weights)
        print("✅ loaded jittor-trained medusa head (strict=True)")

        max_ctx = infer_max_ctx(medusa.base_model, tok, args.max_ctx)

        # 兼容：如果 medusa_generate 不支持 top_p，就自动不传
        import inspect
        sig = inspect.signature(medusa.medusa_generate)
        has_top_p = ("top_p" in sig.parameters)

        def gen_medusa_wrap(prompt, conv_stop_strs, _):
            kwargs = dict(
                temperature=args.temperature,
                max_steps=args.max_new_tokens,   # 你脚本里是 max_steps 的概念
            )
            if has_top_p:
                kwargs["top_p"] = args.top_p

            # 这里复用你 gen_medusa 的逻辑（只要它内部调用 medusa_generate 的参数对上）
            return gen_medusa(
                medusa_model=medusa,
                tok=tok,
                prompt=prompt,
                conv_stop_strs=conv_stop_strs,
                max_ctx=max_ctx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tree_step=args.tree_step,
                tree_pad_steps=args.tree_pad_steps,
                safe_pad=args.safe_pad,
            )

        gen_fn = gen_medusa_wrap

    # Warmup
    for i in range(min(args.warmup, len(qs))):
        q = qs[i]
        prompt, conv = build_prompt(args.conv, [q["turns"][0]], [])
        stop_strs, stop_token_ids = get_conv_stop(conv)
        _ = gen_fn(prompt, stop_strs, stop_token_ids)

    total_time = 0.0
    total_tokens = 0

    # category aggregation
    cat_time = defaultdict(float)
    cat_tokens = defaultdict(int)
    cat_cnt = defaultdict(int)

    pbar = tqdm(qs, desc=f"{args.mode}", unit="q")
    for q in pbar:
        cat = q.get("category", "unknown")
        turns = q["turns"]  # [turn1_user, turn2_user]

        # turn1
        prompt1, conv1 = build_prompt(args.conv, [turns[0]], [])
        stop_strs1, stop_token_ids1 = get_conv_stop(conv1)
        (a1, n1), dt1 = cuda_time(lambda: gen_fn(prompt1, stop_strs1, stop_token_ids1))

        # turn2 (with turn1 assistant reply)
        prompt2, conv2 = build_prompt(args.conv, [turns[0], turns[1]], [a1])
        stop_strs2, stop_token_ids2 = get_conv_stop(conv2)
        (a2, n2), dt2 = cuda_time(lambda: gen_fn(prompt2, stop_strs2, stop_token_ids2))

        dt = dt1 + dt2
        nt = n1 + n2

        total_time += dt
        total_tokens += nt

        cat_time[cat] += dt
        cat_tokens[cat] += nt
        cat_cnt[cat] += 1

        tps = total_tokens / max(total_time, 1e-9)
        pbar.set_postfix(tps=f"{tps:.2f}", dt=f"{dt:.2f}s", cat=cat)

        if args.answer_file and args.model_id:
            append_answer(args.answer_file, q["question_id"], args.model_id, [a1, a2])

    overall_tps = total_tokens / max(total_time, 1e-9)

    per_cat = {}
    for cat, t in cat_time.items():
        tokc = cat_tokens[cat]
        per_cat[cat] = {
            "n_questions": int(cat_cnt[cat]),
            "time_s": float(t),
            "tokens": int(tokc),
            "tokens_per_second": float(tokc / max(t, 1e-9)),
        }

    result = {
        "mode": args.mode,
        "model": args.model,
        "base_model": args.base_model,
        "conv": args.conv,
        "max_ctx": int(max_ctx),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "n_questions": int(len(qs)),
        "total_time_s": float(total_time),
        "total_tokens": int(total_tokens),
        "tokens_per_second": float(overall_tps),
        "per_category": per_cat,
    }

    print(f"mode={args.mode}")
    print(f"model={args.model}")
    if args.base_model:
        print(f"base_model={args.base_model}")
    print(f"conv={args.conv}  max_ctx={max_ctx}")
    print(f"questions={len(qs)}  max_new_tokens={args.max_new_tokens}  temp={args.temperature}  top_p={args.top_p}")
    print(f"total_tokens={total_tokens}  total_time={total_time:.2f}s")
    print(f"tokens_per_second={overall_tps:.2f}")

    # Print per-category tps for convenience
    for cat in sorted(per_cat.keys()):
        print(f"[{cat:12s}] tps={per_cat[cat]['tokens_per_second']:.2f}  "
              f"tokens={per_cat[cat]['tokens']}  time={per_cat[cat]['time_s']:.2f}s  n={per_cat[cat]['n_questions']}")

    if args.out_json:
        save_json(result, args.out_json)


if __name__ == "__main__":
    main()


# # bench_mtbench_jt.py
# # Jittor-only Medusa MT-Bench throughput benchmark (no torch model needed for medusa mode)

# import jittor as jt
# jt.flags.use_cuda = 1

# import argparse, json, time, os
# from collections import defaultdict
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Tuple
# import shortuuid
# import numpy as np
# from tqdm import tqdm

# from transformers import AutoTokenizer, AutoConfig
# from fastchat.conversation import get_conv_template

# # ✅ 确保能 import 你的 repo
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ 这里用你坚持要用的 Jittor 版本
# from medusa_jittor_all.model.medusa_model import MedusaModel


# # ----------------------------
# # IO
# # ----------------------------
# def load_questions(path: str, limit: Optional[int] = None, start: int = 0) -> List[dict]:
#     items = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             items.append(json.loads(line))
#     items = items[start:]
#     if limit is not None:
#         items = items[:limit]
#     return items


# def save_json(obj: Dict[str, Any], path: str):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)


# # ----------------------------
# # Timing helpers (Jittor)
# # ----------------------------
# def jt_time(fn):
#     jt.sync_all(True)
#     t0 = time.time()
#     out = fn()
#     jt.sync_all(True)
#     t1 = time.time()
#     return out, (t1 - t0)


# # ----------------------------
# # Stop strings (FastChat template)
# # ----------------------------
# def get_conv_stop(conv) -> Tuple[List[str], List[int]]:
#     stop_strs = []
#     stop_token_ids = []

#     if hasattr(conv, "stop_str") and conv.stop_str:
#         if isinstance(conv.stop_str, str):
#             stop_strs.append(conv.stop_str)
#         else:
#             stop_strs.extend(list(conv.stop_str))

#     if hasattr(conv, "stop_token_ids") and conv.stop_token_ids:
#         stop_token_ids.extend(list(conv.stop_token_ids))

#     # sep / sep2 sometimes act like stops
#     for name in ["sep", "sep2"]:
#         if hasattr(conv, name):
#             s = getattr(conv, name)
#             if isinstance(s, str) and len(s) > 0 and s not in stop_strs:
#                 if s.strip() != "":
#                     stop_strs.append(s)

#     return stop_strs, stop_token_ids


# def cut_by_stop(text: str, stop_strs: List[str]) -> str:
#     if not stop_strs:
#         return text
#     idxs = [text.find(s) for s in stop_strs if s and text.find(s) != -1]
#     if not idxs:
#         return text
#     cut = min(idxs)
#     return text[:cut]


# # ----------------------------
# # Prompt building (FastChat)
# # ----------------------------
# def build_prompt(conv_name: str, user_messages: List[str], assistant_messages: List[str]) -> Tuple[str, Any]:
#     conv = get_conv_template(conv_name).copy()
#     for i, umsg in enumerate(user_messages):
#         conv.append_message(conv.roles[0], umsg)
#         if i < len(assistant_messages):
#             conv.append_message(conv.roles[1], assistant_messages[i])
#         else:
#             conv.append_message(conv.roles[1], None)
#     return conv.get_prompt(), conv


# # ----------------------------
# # Token counting
# # ----------------------------
# def count_output_tokens(tok, text: str) -> int:
#     return len(tok.encode(text, add_special_tokens=False))


# # ----------------------------
# # Max context inference
# # ----------------------------
# def infer_max_ctx_from_config(cfg, tokenizer, override: Optional[int]) -> int:
#     if override is not None:
#         return int(override)
#     if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings:
#         return int(cfg.max_position_embeddings)
#     m = getattr(tokenizer, "model_max_length", None)
#     if m is not None and isinstance(m, int) and m < 10**6:
#         return int(m)
#     return 2048


# # ----------------------------
# # Medusa head weight loading (Jittor)
# #   Your pkl keys: heads.0.0.linear.weight, heads.0.0.linear.bias, heads.0.1.weight, ...
# #   Jittor medusa_head expected keys: 0.0.linear.weight, 0.0.linear.bias, 0.1.weight, ...
# # ----------------------------
# def load_medusa_head_weights_jt_strict(model, pkl_path: str, cast_fp16: bool = False):
#     sd = jt.load(pkl_path)
#     if not isinstance(sd, dict):
#         raise ValueError(f"jt.load({pkl_path}) did not return a dict, got: {type(sd)}")

#     cleaned = {}
#     for k, v in sd.items():
#         # strip common prefixes
#         if k.startswith("heads."):
#             k = k[len("heads."):]
#         if k.startswith("medusa_head."):
#             k = k[len("medusa_head."):]
#         if k.startswith("module."):
#             k = k[len("module."):]
#         # ensure jt.Var
#         if not isinstance(v, jt.Var):
#             v = jt.array(v)
#         if cast_fp16:
#             v = v.float16()
#         cleaned[k] = v

#     # In Jittor, Module/ModuleList usually has load_state_dict
#     if not hasattr(model, "medusa_head") or not hasattr(model.medusa_head, "load_state_dict"):
#         raise AttributeError("Your Jittor MedusaModel has no medusa_head.load_state_dict; need manual assignment.")

#     # ret = model.medusa_head.load_state_dict(cleaned, strict=True)
#     # # ret could be None or a dict depending on Jittor version
#     # print("✅ Jittor Medusa head weights loaded (strict=True)")
#     ret = model.medusa_head.load_state_dict(cleaned)

#     return ret


# # ----------------------------
# # Generation: medusa (Jittor)
# #   Assumes your MedusaModel.medusa_generate returns token ids (jt.Var) of full sequence.
# # ----------------------------
# def gen_medusa_jt(
#     medusa_model,
#     tok,
#     prompt: str,
#     conv_stop_strs: List[str],
#     max_ctx: int,
#     max_new_tokens: int,
#     temperature: float,
#     top_p: float,
#     tree_step: int = 64,
#     tree_pad_steps: int = 4,
#     safe_pad: int = 32,
# ) -> Tuple[str, int]:
#     # Reserve positions for tree attention
#     tree_pad = tree_pad_steps * tree_step + safe_pad

#     # leave small safety
#     max_in = max(1, max_ctx - tree_pad - 16)

#     enc = tok(prompt, return_tensors="np", truncation=True, max_length=max_in)
#     input_ids_np = enc["input_ids"].astype(np.int32)  # embedding usually wants int32
#     input_ids = jt.array(input_ids_np)

#     in_len = int(input_ids_np.shape[1])

#     # Cap steps so we don't exceed ctx
#     max_steps_allow = max(1, min(max_new_tokens, max_ctx - tree_pad - in_len))

#     out_ids = medusa_model.medusa_generate(
#         input_ids,
#         temperature=temperature,
#         top_p=top_p,
#         max_steps=max_steps_allow,
#     )

#     # out_ids: jt.Var [1, total_len]
#     out_np = out_ids.numpy()
#     gen_ids = out_np[0, in_len:].tolist()

#     out_text = tok.decode(gen_ids, skip_special_tokens=True)
#     out_text = cut_by_stop(out_text, conv_stop_strs).strip()
#     n_out = count_output_tokens(tok, out_text)
#     return out_text, n_out


# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--questions", type=str, required=True)
#     ap.add_argument("--mode", choices=["medusa"], required=True, help="Only medusa (Jittor) is supported in this script")
#     ap.add_argument("--model", type=str, required=True, help="Base model path (for config/tokenizer)")
#     ap.add_argument("--conv", type=str, default="vicuna_v1.1")

#     ap.add_argument("--max_new_tokens", type=int, default=256)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_p", type=float, default=0.95)

#     ap.add_argument("--limit", type=int, default=None)
#     ap.add_argument("--start", type=int, default=0)
#     ap.add_argument("--warmup", type=int, default=3)

#     ap.add_argument("--max_ctx", type=int, default=None)
#     ap.add_argument("--out_json", type=str, default=None)

#     ap.add_argument("--tree_step", type=int, default=64)
#     ap.add_argument("--tree_pad_steps", type=int, default=4)
#     ap.add_argument("--safe_pad", type=int, default=32)

#     ap.add_argument("--answer_file", type=str, default=None)
#     ap.add_argument("--model_id", type=str, default=None)

#     ap.add_argument("--medusa_weights", type=str, required=True, help="Path to jittor-trained medusa head .pkl")
#     ap.add_argument("--medusa_num_heads", type=int, default=3)
#     ap.add_argument("--medusa_num_layers", type=int, default=1)
#     ap.add_argument("--cast_fp16", action="store_true", help="Cast loaded head weights to fp16")

#     args = ap.parse_args()

#     def append_answer(answer_file, question_id, model_id, turns):
#         os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#         ans_json = {
#             "question_id": question_id,
#             "answer_id": shortuuid.uuid(),
#             "model_id": model_id,
#             "choices": [{"index": 0, "turns": turns}],
#             "tstamp": time.time(),
#         }
#         with open(answer_file, "a", encoding="utf-8") as f:
#             f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

#     qs = load_questions(args.questions, limit=args.limit, start=args.start)
#     assert len(qs) > 0, "no questions loaded"

#     # Tokenizer + config
#     tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
#     cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
#     cfg.medusa_num_heads = int(args.medusa_num_heads)
#     cfg.medusa_num_layers = int(args.medusa_num_layers)

#     # Build Jittor medusa model
#     medusa = MedusaModel.from_pretrained(args.model, config=cfg)
#     medusa.eval()

#     # Load head weights (.pkl)
#     load_medusa_head_weights_jt_strict(medusa, args.medusa_weights, cast_fp16=args.cast_fp16)

#     max_ctx = infer_max_ctx_from_config(cfg, tok, args.max_ctx)

#     # Warmup
#     for i in range(min(args.warmup, len(qs))):
#         q = qs[i]
#         prompt, conv = build_prompt(args.conv, [q["turns"][0]], [])
#         stop_strs, _ = get_conv_stop(conv)
#         _ = gen_medusa_jt(
#             medusa_model=medusa,
#             tok=tok,
#             prompt=prompt,
#             conv_stop_strs=stop_strs,
#             max_ctx=max_ctx,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             tree_step=args.tree_step,
#             tree_pad_steps=args.tree_pad_steps,
#             safe_pad=args.safe_pad,
#         )

#     total_time = 0.0
#     total_tokens = 0

#     cat_time = defaultdict(float)
#     cat_tokens = defaultdict(int)
#     cat_cnt = defaultdict(int)

#     pbar = tqdm(qs, desc=f"medusa_jt", unit="q")
#     for q in pbar:
#         cat = q.get("category", "unknown")
#         turns = q["turns"]

#         # turn1
#         prompt1, conv1 = build_prompt(args.conv, [turns[0]], [])
#         stop_strs1, _ = get_conv_stop(conv1)
#         (a1, n1), dt1 = jt_time(lambda: gen_medusa_jt(
#             medusa_model=medusa,
#             tok=tok,
#             prompt=prompt1,
#             conv_stop_strs=stop_strs1,
#             max_ctx=max_ctx,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             tree_step=args.tree_step,
#             tree_pad_steps=args.tree_pad_steps,
#             safe_pad=args.safe_pad,
#         ))

#         # turn2
#         prompt2, conv2 = build_prompt(args.conv, [turns[0], turns[1]], [a1])
#         stop_strs2, _ = get_conv_stop(conv2)
#         (a2, n2), dt2 = jt_time(lambda: gen_medusa_jt(
#             medusa_model=medusa,
#             tok=tok,
#             prompt=prompt2,
#             conv_stop_strs=stop_strs2,
#             max_ctx=max_ctx,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             tree_step=args.tree_step,
#             tree_pad_steps=args.tree_pad_steps,
#             safe_pad=args.safe_pad,
#         ))

#         dt = dt1 + dt2
#         nt = n1 + n2

#         total_time += dt
#         total_tokens += nt

#         cat_time[cat] += dt
#         cat_tokens[cat] += nt
#         cat_cnt[cat] += 1

#         tps = total_tokens / max(total_time, 1e-9)
#         pbar.set_postfix(tps=f"{tps:.2f}", dt=f"{dt:.2f}s", cat=cat)

#         if args.answer_file and args.model_id:
#             append_answer(args.answer_file, q["question_id"], args.model_id, [a1, a2])

#     overall_tps = total_tokens / max(total_time, 1e-9)

#     per_cat = {}
#     for cat, t in cat_time.items():
#         tokc = cat_tokens[cat]
#         per_cat[cat] = {
#             "n_questions": int(cat_cnt[cat]),
#             "time_s": float(t),
#             "tokens": int(tokc),
#             "tokens_per_second": float(tokc / max(t, 1e-9)),
#         }

#     result = {
#         "mode": "medusa_jt",
#         "model": args.model,
#         "conv": args.conv,
#         "max_ctx": int(max_ctx),
#         "max_new_tokens": int(args.max_new_tokens),
#         "temperature": float(args.temperature),
#         "top_p": float(args.top_p),
#         "n_questions": int(len(qs)),
#         "total_time_s": float(total_time),
#         "total_tokens": int(total_tokens),
#         "tokens_per_second": float(overall_tps),
#         "per_category": per_cat,
#         "medusa_weights": args.medusa_weights,
#         "medusa_num_heads": int(args.medusa_num_heads),
#         "medusa_num_layers": int(args.medusa_num_layers),
#     }

#     print(f"mode=medusa_jt")
#     print(f"model={args.model}")
#     print(f"conv={args.conv}  max_ctx={max_ctx}")
#     print(f"questions={len(qs)}  max_new_tokens={args.max_new_tokens}  temp={args.temperature}  top_p={args.top_p}")
#     print(f"total_tokens={total_tokens}  total_time={total_time:.2f}s")
#     print(f"tokens_per_second={overall_tps:.2f}")

#     for cat in sorted(per_cat.keys()):
#         print(f"[{cat:12s}] tps={per_cat[cat]['tokens_per_second']:.2f}  "
#               f"tokens={per_cat[cat]['tokens']}  time={per_cat[cat]['time_s']:.2f}s  n={per_cat[cat]['n_questions']}")

#     if args.out_json:
#         save_json(result, args.out_json)


# if __name__ == "__main__":
#     main()
