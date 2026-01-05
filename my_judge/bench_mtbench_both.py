#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT-Bench throughput benchmark (one script):
- baseline: HF/torch generate
- medusa: Jedusa (jittor) medusa_generate

Features:
- --run baseline/medusa/both
- saves:
  * console log (tee): --log_file
  * merged summary json: --out_json (for --run both it will NOT be overwritten; we merge stage jsons)
  * per-question detail jsonl: --detail_jsonl
  * FastChat judge answer jsonl: --baseline_answer_file / --medusa_answer_file

Key fix for your "medusa tokens drop":
- Medusa prompt truncation now reserves the SAME output budget as baseline:
    max_in = max_ctx - max_new_tokens - tree_pad - 32
  so medusa isn't starved of input.
- Also caps max_steps_allow with KV headroom for tree decoding safety.
"""

import argparse, json, time, os, sys, inspect, gc, subprocess, io
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import shortuuid
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from fastchat.conversation import get_conv_template


# ----------------------------
# utils
# ----------------------------
def now_ts():
    return time.strftime("%H:%M:%S")

def dprint(enabled: bool, msg: str):
    if enabled:
        print(msg, flush=True)

def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_jsonl(path: str, obj: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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

def build_prompt(conv_name: str, user_messages: List[str], assistant_messages: List[str]):
    conv = get_conv_template(conv_name).copy()
    for i, umsg in enumerate(user_messages):
        conv.append_message(conv.roles[0], umsg)
        if i < len(assistant_messages):
            conv.append_message(conv.roles[1], assistant_messages[i])
        else:
            conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv

def get_conv_stop(conv) -> Tuple[List[str], List[int]]:
    stop_strs = []
    stop_token_ids = []

    if hasattr(conv, "stop_str") and conv.stop_str:
        if isinstance(conv.stop_str, str):
            stop_strs.append(conv.stop_str)
        else:
            stop_strs.extend(list(conv.stop_str))

    if hasattr(conv, "stop_token_ids") and conv.stop_token_ids:
        stop_token_ids.extend(list(conv.stop_token_ids))

    for name in ["sep", "sep2"]:
        if hasattr(conv, name):
            s = getattr(conv, name)
            if isinstance(s, str) and len(s) > 0 and s not in stop_strs:
                if s.strip() != "":
                    stop_strs.append(s)
    return stop_strs, stop_token_ids

def cut_by_stop(text: str, stop_strs: List[str]) -> str:
    if not stop_strs:
        return text
    idxs = [text.find(s) for s in stop_strs if s and text.find(s) != -1]
    if not idxs:
        return text
    return text[:min(idxs)]

def count_output_tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def infer_max_ctx(model, tokenizer, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    if hasattr(model, "config") and getattr(model.config, "max_position_embeddings", None):
        return int(model.config.max_position_embeddings)
    m = getattr(tokenizer, "model_max_length", None)
    if m is not None and isinstance(m, int) and m < 10**6:
        return int(m)
    return 2048


# ----------------------------
# logging tee
# ----------------------------
class Tee(io.TextIOBase):
    """Duplicate stdout/stderr to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

def setup_logging(log_file: Optional[str], tag: str) -> Optional[io.TextIOWrapper]:
    """If log_file provided, tee stdout/stderr to it (append mode)."""
    if not log_file:
        return None
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "a", encoding="utf-8")
    f.write(f"\n\n===== {tag} START {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    f.flush()
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    return f


# ----------------------------
# timing helpers
# ----------------------------
def _try_import_jittor():
    try:
        import jittor as jt
        return jt
    except ImportError:
        return None

def sync_torch():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def sync_jittor():
    jt = _try_import_jittor()
    if jt is None:
        return
    try:
        if getattr(jt.flags, "use_cuda", 0):
            jt.sync_all(True)
        else:
            jt.sync_all()
    except Exception:
        pass

def time_torch(fn):
    sync_torch()
    t0 = time.time()
    out = fn()
    sync_torch()
    return out, (time.time() - t0)

def time_jittor(fn):
    sync_jittor()
    t0 = time.time()
    out = fn()
    sync_jittor()
    return out, (time.time() - t0)

def cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cleanup_jittor():
    jt = _try_import_jittor()
    if jt is None:
        return
    try:
        jt.sync_all(True)
    except Exception:
        pass
    for name in ["gc", "clean_cache", "clear_cache"]:
        if hasattr(jt, name) and callable(getattr(jt, name)):
            try:
                getattr(jt, name)()
            except Exception:
                pass


# ----------------------------
# baseline stop criteria (HF torch)
# ----------------------------
class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs: List[str], start_len: int, window: int = 64):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strs = [s for s in stop_strs if s]
        self.start_len = start_len
        self.window = window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_strs:
            return False
        gen = input_ids[0, self.start_len:]
        if gen.numel() <= 0:
            return False
        tail = gen[-self.window:].tolist()
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return any(s in text for s in self.stop_strs)


# ----------------------------
# baseline generation (HF torch)
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
    safety = 16
    max_in = max(1, max_ctx - max_new_tokens - safety)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    do_sample = temperature > 0
    eos_ids = []
    if model.config.eos_token_id is not None:
        eos_ids.append(int(model.config.eos_token_id))
    eos_ids.extend([int(x) for x in (conv_stop_token_ids or []) if x is not None])

    start_len = inputs["input_ids"].shape[1]
    stop_criteria = StoppingCriteriaList()
    if conv_stop_strs:
        stop_criteria.append(StopOnSubstrings(tok, conv_stop_strs, start_len=start_len, window=64))

    gen_kwargs = dict(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=model.config.eos_token_id,
        stopping_criteria=stop_criteria if len(stop_criteria) > 0 else None,
    )

    if len(eos_ids) == 1:
        gen_kwargs["eos_token_id"] = eos_ids[0]
    elif len(eos_ids) > 1:
        try:
            gen_kwargs["eos_token_id"] = eos_ids
        except Exception:
            gen_kwargs["eos_token_id"] = eos_ids[0]

    out_ids = model.generate(**gen_kwargs)
    gen_ids = out_ids[0, start_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    text = cut_by_stop(text, conv_stop_strs).strip()
    n_out = count_output_tokens(tok, text)
    return text, n_out


# ----------------------------
# medusa helpers (Jittor)
# ----------------------------
def _decode_from_token_ids(tok, token_ids: np.ndarray, start_len: int, conv_stop_strs: List[str]) -> Tuple[str, int]:
    if token_ids.ndim == 2:
        ids = token_ids[0]
    else:
        ids = token_ids
    gen = ids[start_len:]
    text = tok.decode(gen.tolist(), skip_special_tokens=True)
    text = cut_by_stop(text, conv_stop_strs).strip()
    return text, count_output_tokens(tok, text)

def _pick_max_steps_param(sig: inspect.Signature) -> Optional[str]:
    for name in ["max_steps", "max_new_tokens", "max_tokens", "steps", "max_length"]:
        if name in sig.parameters:
            return name
    return None

def load_medusa_head_weights_best_effort(medusa_model, weights_path: str, cast_fp16: bool = False, debug: bool = False):
    if not weights_path:
        return

    if hasattr(medusa_model, "load_medusa_head") and callable(getattr(medusa_model, "load_medusa_head")):
        dprint(debug, f"[medusa-debug] {now_ts()} load_medusa_head({weights_path})")
        medusa_model.load_medusa_head(weights_path)
        print("✅ loaded medusa head via model.load_medusa_head()", flush=True)
        return

    jt = _try_import_jittor()
    if jt is None:
        raise RuntimeError("Need jittor to load .pkl medusa head weights, but jittor import failed.")

    sd = jt.load(weights_path)
    if not isinstance(sd, dict):
        raise ValueError(f"jt.load({weights_path}) did not return a dict, got {type(sd)}")

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("heads."):
            k = k[len("heads."):]
        if k.startswith("medusa_head."):
            k = k[len("medusa_head."):]
        if k.startswith("module."):
            k = k[len("module."):]
        if not isinstance(v, jt.Var):
            v = jt.array(v)
        if cast_fp16:
            v = v.float16()
        cleaned[k] = v

    if not hasattr(medusa_model, "medusa_head") or not hasattr(medusa_model.medusa_head, "load_state_dict"):
        raise AttributeError("Model has no medusa_head.load_state_dict; cannot auto-load head weights.")

    medusa_model.medusa_head.load_state_dict(cleaned)
    print("✅ loaded medusa head via medusa_head.load_state_dict()", flush=True)

def gen_medusa(
    medusa_model,
    tok,
    prompt: str,
    conv_stop_strs: List[str],
    max_ctx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tree_step: int = 64,
    tree_pad_steps: int = 4,
    safe_pad: int = 32,
    debug: bool = False,
) -> Tuple[str, int]:
    jt = _try_import_jittor()
    if jt is None:
        raise RuntimeError("medusa mode requires jittor, but jittor import failed.")

    tree_pad = int(tree_pad_steps) * int(tree_step) + int(safe_pad)

    # ✅ IMPORTANT FIX: reserve max_new_tokens budget just like baseline
    # so medusa isn't starved of prompt space.
    max_in = max(1, int(max_ctx) - int(max_new_tokens) - int(tree_pad) - 32)

    # tokenizer in torch is fine (CPU), then to numpy->jt
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)
    pt_ids = enc["input_ids"]
    in_len = int(pt_ids.shape[1])

    np_ids = np.asarray(pt_ids.numpy(), dtype=np.int32)
    input_ids = jt.array(np_ids)

    # tree decoding temporary KV headroom (avoid boundary append overflow)
    kv_headroom = int(tree_step) + 16
    max_steps_allow = max(
        1,
        min(
            int(max_new_tokens),
            int(max_ctx) - int(tree_pad) - in_len - kv_headroom
        ),
    )

    sig = inspect.signature(medusa_model.medusa_generate)
    max_steps_name = _pick_max_steps_param(sig)

    kwargs = {"temperature": float(temperature)}
    if "top_p" in sig.parameters:
        kwargs["top_p"] = float(top_p)
    
    # Force debug param (we know we added it)
    kwargs["debug"] = debug

    if max_steps_name is not None:
        kwargs[max_steps_name] = int(max_steps_allow)
    else:
        kwargs["max_steps"] = int(max_steps_allow)

    dprint(
        debug,
        "[medusa-debug] budget "
        f"max_ctx={max_ctx} "
        f"tree_pad={tree_pad} tree_step={tree_step} kv_headroom={kv_headroom} "
        f"max_in={max_in} in_len={in_len} max_new_tokens={max_new_tokens} max_steps_allow={max_steps_allow}",
    )
    dprint(debug, f"[medusa-debug] {now_ts()} medusa_generate kwargs={kwargs}")

    with jt.no_grad():
        out = medusa_model.medusa_generate(input_ids, **kwargs)

    if isinstance(out, jt.Var):
        out_np = out.numpy()
        out_len = int(out_np.shape[1]) if out_np.ndim == 2 else int(out_np.reshape(1, -1).shape[1])
        dprint(debug, f"[medusa-debug] out_ids: out_len={out_len} in_len={in_len} gen_tokens={max(0, out_len - in_len)}")
        text, n_out = _decode_from_token_ids(tok, out_np, start_len=in_len, conv_stop_strs=conv_stop_strs)
        dprint(debug, f"[medusa-debug] decoded: n_out={n_out} text_preview={text[:80]!r}")
        return text, n_out

    if isinstance(out, str):
        out_text = cut_by_stop(out, conv_stop_strs).strip()
        return out_text, count_output_tokens(tok, out_text)

    if hasattr(out, "__iter__"):
        deltas = []
        last_text = ""
        for item in out:
            if isinstance(item, str):
                deltas.append(item)
            elif isinstance(item, dict):
                if item.get("delta"):
                    deltas.append(item["delta"])
                elif item.get("text"):
                    last_text = item["text"]
                elif item.get("content"):
                    last_text = item["content"]
            else:
                try:
                    if isinstance(item, jt.Var):
                        np_ids2 = item.numpy().reshape(-1)
                        deltas.append(tok.decode(np_ids2.tolist(), skip_special_tokens=True))
                except Exception:
                    pass
        out_text = "".join(deltas) if deltas else last_text
        out_text = cut_by_stop(out_text, conv_stop_strs).strip()
        return out_text, count_output_tokens(tok, out_text)

    return "", 0


# ----------------------------
# jittor baseline generation (single vicuna, no medusa heads)
# ----------------------------
def _jt_argmax_indices(x, dim: int = -1):
    """Jittor argmax may return Var or (indices, values); normalize to indices Var."""
    jt = _try_import_jittor()
    if jt is None:
        raise RuntimeError("Need jittor for _jt_argmax_indices, but jittor import failed.")

    r = jt.argmax(x, dim=dim)
    if isinstance(r, (tuple, list)) and len(r) == 2:
        a, b = r
        # Prefer integer dtype output as indices.
        try:
            a_is_int = str(a.dtype).startswith("int")
        except Exception:
            a_is_int = False
        try:
            b_is_int = str(b.dtype).startswith("int")
        except Exception:
            b_is_int = False
        if a_is_int and (not b_is_int):
            return a
        if b_is_int and (not a_is_int):
            return b
        # Fallback: first element is usually indices in this repo.
        return a
    return r


def gen_jt_baseline(
    jt_model,
    tok,
    prompt: str,
    conv_stop_strs: List[str],
    conv_stop_token_ids: List[int],
    max_ctx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    debug: bool = False,
) -> Tuple[str, int]:
    """Jittor-only Vicuna decode (KV-cache greedy). Temperature/top_p are accepted but not used."""
    jt = _try_import_jittor()
    if jt is None:
        raise RuntimeError("jt_baseline mode requires jittor, but jittor import failed.")

    if temperature and float(temperature) > 0:
        dprint(debug, "[jt-baseline] NOTE: using greedy decode (temperature/top_p ignored).")

    safety = 16
    max_in = max(1, int(max_ctx) - int(max_new_tokens) - safety)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)
    pt_ids = enc["input_ids"]
    in_len = int(pt_ids.shape[1])
    input_ids = jt.array(np.asarray(pt_ids.numpy(), dtype=np.int32))

    # KV cache with fixed cap (PyTorch-style)
    try:
        from medusa_jittor_all.model.kv_cache import initialize_past_key_values
    except Exception as e:
        raise RuntimeError(f"Failed to import initialize_past_key_values for jt_baseline: {e}")

    cap = int(max_ctx)
    past_key_values, _pkv_data, _len_data = initialize_past_key_values(
        jt_model,
        max_cache_len=cap,
        dtype=jt.float16,
    )

    # Prefill
    with jt.no_grad():
        out = jt_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    logits = getattr(out, "logits", out[0] if isinstance(out, (tuple, list)) else None)
    if logits is None:
        raise RuntimeError("jt_baseline: model forward did not return logits")

    eos_ids = set(int(x) for x in (conv_stop_token_ids or []) if x is not None)
    try:
        if getattr(jt_model.config, "eos_token_id", None) is not None:
            eos_ids.add(int(jt_model.config.eos_token_id))
    except Exception:
        pass

    cur = input_ids
    max_steps = int(max_new_tokens)

    for step in range(max_steps):
        next_id = _jt_argmax_indices(logits[:, -1, :], dim=-1).cast(jt.int32)  # [1]
        # Append for final decode (kept on device; single sync only if we early-stop)
        next_tok = next_id.reshape(1, 1)
        cur = jt.concat([cur, next_tok], dim=1)

        if eos_ids:
            try:
                nid = int(next_id.item())
                if nid in eos_ids:
                    break
            except Exception:
                pass

        with jt.no_grad():
            out = jt_model(input_ids=next_tok, past_key_values=past_key_values, use_cache=True)
        logits = getattr(out, "logits", out[0] if isinstance(out, (tuple, list)) else None)
        if logits is None:
            raise RuntimeError("jt_baseline: step forward did not return logits")

    out_np = cur.numpy()
    text, n_out = _decode_from_token_ids(tok, out_np, start_len=in_len, conv_stop_strs=conv_stop_strs)
    return text, n_out


# ----------------------------
# FastChat judge answer writer
# ----------------------------
def append_answer(answer_file: str, question_id: str, model_id: str, turns: List[str]):
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


# ----------------------------
# Medusa dynamic import resolver
# ----------------------------
def resolve_medusa_class(spec: str):
    cls_name = None
    mod_name = spec

    if ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
    else:
        parts = spec.split(".")
        if parts and parts[-1] and parts[-1][0].isupper():
            mod_name = ".".join(parts[:-1])
            cls_name = parts[-1]

    mod = __import__(mod_name, fromlist=["*"])

    if cls_name is not None:
        obj = getattr(mod, cls_name, None)
        if obj is None:
            raise AttributeError(f"Module '{mod_name}' has no attribute '{cls_name}'.")
        return obj

    if hasattr(mod, "MedusaModel"):
        return getattr(mod, "MedusaModel")

    for k in dir(mod):
        if k.lower() == "medusamodel" or k.endswith("MedusaModel"):
            return getattr(mod, k)

    raise AttributeError(
        f"Cannot find MedusaModel class in module '{mod_name}'. "
        f"Please pass --medusa_impl like '{mod_name}:MedusaModel'."
    )


# ----------------------------
# benchmark runner (per mode)
# ----------------------------
def run_benchmark(
    mode: str,
    qs: List[dict],
    tok,
    conv_name: str,
    warmup: int,
    gen_turn12_fn,
    time_fn,
    answer_file: Optional[str],
    model_id: Optional[str],
    detail_jsonl: Optional[str] = None,
):
    warm_n = min(warmup, len(qs))
    print(f"\nStart Warmup [{mode}] (n={warm_n})...", flush=True)
    for i in range(warm_n):
        q = qs[i]
        prompt, conv = build_prompt(conv_name, [q["turns"][0]], [])
        stop_strs, _ = get_conv_stop(conv)
        _ = gen_turn12_fn(prompt, stop_strs, [], warmup_only=True)
    print(f"Warmup [{mode}] done. Starting benchmark...", flush=True)

    total_time = 0.0
    total_tokens = 0
    cat_time = defaultdict(float)
    cat_tokens = defaultdict(int)
    cat_cnt = defaultdict(int)

    pbar = tqdm(qs, desc=mode, unit="q")
    for q in pbar:
        cat = q.get("category", "unknown")
        turns = q["turns"]

        def _one_q():
            prompt1, conv1 = build_prompt(conv_name, [turns[0]], [])
            stop_strs1, stop_token_ids1 = get_conv_stop(conv1)
            a1, a2, n1, n2 = gen_turn12_fn(
                prompt1, stop_strs1, stop_token_ids1, warmup_only=False, turns=turns
            )
            return a1, a2, n1, n2

        (a1, a2, n1, n2), dt = time_fn(_one_q)
        nt = int(n1) + int(n2)

        total_time += dt
        total_tokens += nt
        cat_time[cat] += dt
        cat_tokens[cat] += nt
        cat_cnt[cat] += 1

        tps = total_tokens / max(total_time, 1e-9)
        pbar.set_postfix(tps=f"{tps:.2f}", dt=f"{dt:.2f}s", cat=cat)

        if answer_file and model_id:
            append_answer(answer_file, q["question_id"], model_id, [a1, a2])

        if detail_jsonl:
            append_jsonl(detail_jsonl, {
                "mode": mode,
                "question_id": q.get("question_id"),
                "category": cat,
                "turn1": {"tokens": int(n1), "text": a1},
                "turn2": {"tokens": int(n2), "text": a2},
                "total": {"time_s": float(dt), "tokens": int(nt), "tps": float(nt / max(dt, 1e-9))},
            })

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

    return {
        "mode": mode,
        "n_questions": int(len(qs)),
        "total_time_s": float(total_time),
        "total_tokens": int(total_tokens),
        "tokens_per_second": float(overall_tps),
        "per_category": per_cat,
    }


# ----------------------------
# argv helper for subprocess
# ----------------------------
def replace_or_add_arg(argv: List[str], key: str, value: Optional[str]) -> List[str]:
    """Replace `--key <val>` if exists; otherwise append. If value is None, remove the arg (and its value if present)."""
    out = []
    i = 0
    found = False
    while i < len(argv):
        a = argv[i]
        if a == key:
            found = True
            # skip this key and its value (if any)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            if value is not None:
                out += [key, value]
            continue
        out.append(a)
        i += 1
    if (not found) and (value is not None):
        out += [key, value]
    return out

def replace_run(argv: List[str], new_run: str) -> List[str]:
    argv2 = replace_or_add_arg(argv, "--run", new_run)
    return argv2


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=str, required=True)
    ap.add_argument("--conv", type=str, default="vicuna_v1.1")
    ap.add_argument("--run", choices=["baseline", "medusa", "jt_baseline", "both"], default="both")

    ap.add_argument("--baseline_model", type=str, required=True)
    ap.add_argument("--medusa_model", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default=None)

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max_ctx", type=int, default=None)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--detail_jsonl", type=str, default=None,
                    help="Append per-question details (mode/text/tokens/time) as jsonl")
    ap.add_argument("--log_file", type=str, default=None,
                    help="Append console output to this file (stdout+stderr)")

    # medusa
    ap.add_argument("--jedusa_path", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--medusa_impl", type=str, default="medusa_jittor_all.model.medusa_model:MedusaModel")
    ap.add_argument("--medusa_weights", type=str, default=None)
    ap.add_argument("--medusa_num_heads", type=int, default=3)
    ap.add_argument("--cast_fp16_head", action="store_true")

    ap.add_argument("--tree_step", type=int, default=64)
    ap.add_argument("--tree_pad_steps", type=int, default=4)
    ap.add_argument("--safe_pad", type=int, default=32)

    # kv cache cap (optional)
    ap.add_argument("--kv_cache_len", type=int, default=None,
                    help="Forward initialize_past_key_values(max_cache_len=...) inside medusa_jittor_all if available.")

    # answers
    ap.add_argument("--baseline_answer_file", type=str, default=None)
    ap.add_argument("--medusa_answer_file", type=str, default=None)
    ap.add_argument("--jt_answer_file", type=str, default=None)
    ap.add_argument("--baseline_model_id", type=str, default=None)
    ap.add_argument("--medusa_model_id", type=str, default=None)
    ap.add_argument("--jt_model_id", type=str, default=None)

    # debug
    ap.add_argument("--debug_baseline", action="store_true")
    ap.add_argument("--debug_medusa", action="store_true")

    args = ap.parse_args()

    # tee logs for single-stage runs; in "both", subprocs will tee independently.
    log_handle = None
    if args.run in ["baseline", "medusa", "jt_baseline"]:
        log_handle = setup_logging(args.log_file, tag=args.run)

    qs = load_questions(args.questions, limit=args.limit, start=args.start)
    assert len(qs) > 0, "no questions loaded"

    tok_path = args.tokenizer or args.baseline_model
    tok = AutoTokenizer.from_pretrained(tok_path, use_fast=False)

    combined = {
        "run": args.run,
        "conv": args.conv,
        "questions": len(qs),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "results": {}
    }

    # ==========================================================
    # --run both: run medusa then baseline as subprocess
    # and merge stage json into args.out_json.
    # ==========================================================
    if args.run == "both":
        argv = sys.argv[1:]
        base_cmd = [sys.executable, os.path.abspath(__file__)]
        env = os.environ.copy()

        stage_out_medusa = None
        stage_out_baseline = None
        if args.out_json:
            p = Path(args.out_json)
            stage_out_medusa = str(p.with_suffix(p.suffix + ".medusa.json"))
            stage_out_baseline = str(p.with_suffix(p.suffix + ".baseline.json"))

        # medusa subprocess
        argv_m = replace_run(argv, "medusa")
        if stage_out_medusa:
            argv_m = replace_or_add_arg(argv_m, "--out_json", stage_out_medusa)
        r1 = subprocess.run(base_cmd + argv_m, env=env)
        if r1.returncode != 0:
            raise SystemExit(r1.returncode)

        # baseline subprocess
        argv_b = replace_run(argv, "baseline")
        if stage_out_baseline:
            argv_b = replace_or_add_arg(argv_b, "--out_json", stage_out_baseline)
        r2 = subprocess.run(base_cmd + argv_b, env=env)
        if r2.returncode != 0:
            raise SystemExit(r2.returncode)

        # merge stage jsons
        if args.out_json and stage_out_medusa and stage_out_baseline:
            try:
                jm = load_json(stage_out_medusa)
                jb = load_json(stage_out_baseline)
                merged = {
                    "run": "both",
                    "conv": args.conv,
                    "questions": len(qs),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "results": {}
                }
                # stage files are "combined-like"; keep their results if present
                if isinstance(jm, dict) and "results" in jm and "medusa" in jm["results"]:
                    merged["results"]["medusa"] = jm["results"]["medusa"]
                elif isinstance(jm, dict) and "results" in jm and len(jm["results"]) == 1:
                    merged["results"].update(jm["results"])

                if isinstance(jb, dict) and "results" in jb and "baseline" in jb["results"]:
                    merged["results"]["baseline"] = jb["results"]["baseline"]
                elif isinstance(jb, dict) and "results" in jb and len(jb["results"]) == 1:
                    merged["results"].update(jb["results"])

                save_json(merged, args.out_json)
                print(f"\n✅ merged json written to: {args.out_json}", flush=True)
            except Exception as e:
                print(f"[WARN] failed to merge stage jsons: {e}", flush=True)

        return

    # ==========================================================
    # single-stage runs: medusa or baseline
    # ==========================================================
    if args.run == "medusa":
        if args.jedusa_path and args.jedusa_path not in sys.path:
            sys.path.append(args.jedusa_path)

        jt = _try_import_jittor()
        if jt is None:
            raise RuntimeError("Requested medusa run, but jittor import failed.")
        try:
            jt.flags.use_cuda = 1
        except Exception:
            pass

        MedusaModel = resolve_medusa_class(args.medusa_impl)

        dprint(args.debug_medusa, f"[medusa-debug] {now_ts()} loading medusa base: {args.medusa_model}")
        if hasattr(MedusaModel, "from_pretrained") and callable(getattr(MedusaModel, "from_pretrained")):
            sig = inspect.signature(MedusaModel.from_pretrained)
            kw = {}
            for k, v in {
                "medusa_num_heads": args.medusa_num_heads,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }.items():
                if k in sig.parameters:
                    kw[k] = v
            medusa_model = MedusaModel.from_pretrained(args.medusa_model, **kw)
        else:
            medusa_model = MedusaModel(args.medusa_model)

        if hasattr(medusa_model, "eval"):
            medusa_model.eval()

        if args.medusa_weights:
            load_medusa_head_weights_best_effort(
                medusa_model,
                args.medusa_weights,
                cast_fp16=args.cast_fp16_head,
                debug=args.debug_medusa,
            )

        # infer ctx from model config, cap to model's own max_position_embeddings if present
        if hasattr(medusa_model, "base_model"):
            medusa_max_ctx = infer_max_ctx(medusa_model.base_model, tok, args.max_ctx)
        else:
            medusa_max_ctx = infer_max_ctx(medusa_model, tok, args.max_ctx)

        cap_ctx = None
        try:
            if hasattr(medusa_model, "config") and getattr(medusa_model.config, "max_position_embeddings", None):
                cap_ctx = int(medusa_model.config.max_position_embeddings)
            elif hasattr(medusa_model, "base_model") and hasattr(medusa_model.base_model, "config") and getattr(medusa_model.base_model.config, "max_position_embeddings", None):
                cap_ctx = int(medusa_model.base_model.config.max_position_embeddings)
        except Exception:
            cap_ctx = None

        if cap_ctx is not None:
            medusa_max_ctx = int(min(int(medusa_max_ctx), int(cap_ctx)))

        dprint(args.debug_medusa, f"[medusa-debug] inferred_ctx={medusa_max_ctx} cap_ctx={cap_ctx}")

        # Optional kv_cache patch
        if args.kv_cache_len is not None:
            try:
                import medusa_jittor_all.model.kv_cache as kv_cache
                _orig_init = kv_cache.initialize_past_key_values

                def _wrapped_init(model, *a, **kw):
                    kw["max_cache_len"] = int(args.kv_cache_len)
                    return _orig_init(model, *a, **kw)

                kv_cache.initialize_past_key_values = _wrapped_init
                dprint(args.debug_medusa, f"[medusa-debug] patched initialize_past_key_values(max_cache_len={args.kv_cache_len})")
            except Exception as e:
                print(f"[WARN] kv_cache patch failed. err={e}", flush=True)

        def medusa_turn12(prompt1, stop_strs1, _stop_token_ids_unused, warmup_only=False, turns=None):
            a1, n1 = gen_medusa(
                medusa_model, tok, prompt1, stop_strs1,
                medusa_max_ctx, args.max_new_tokens, args.temperature, args.top_p,
                tree_step=args.tree_step, tree_pad_steps=args.tree_pad_steps, safe_pad=args.safe_pad,
                debug=args.debug_medusa
            )
            if warmup_only:
                return a1, "", n1, 0

            prompt2, conv2 = build_prompt(args.conv, [turns[0], turns[1]], [a1])
            stop_strs2, _ = get_conv_stop(conv2)
            a2, n2 = gen_medusa(
                medusa_model, tok, prompt2, stop_strs2,
                medusa_max_ctx, args.max_new_tokens, args.temperature, args.top_p,
                tree_step=args.tree_step, tree_pad_steps=args.tree_pad_steps, safe_pad=args.safe_pad,
                debug=args.debug_medusa
            )
            return a1, a2, n1, n2

        res_m = run_benchmark(
            mode="medusa",
            qs=qs,
            tok=tok,
            conv_name=args.conv,
            warmup=args.warmup,
            gen_turn12_fn=medusa_turn12,
            time_fn=time_jittor,
            answer_file=args.medusa_answer_file,
            model_id=args.medusa_model_id,
            detail_jsonl=args.detail_jsonl,
        )
        res_m["model"] = args.medusa_model
        res_m["max_ctx"] = int(medusa_max_ctx)
        res_m["kv_cache_len"] = (int(args.kv_cache_len) if args.kv_cache_len is not None else None)
        combined["results"]["medusa"] = res_m

        print("\n=== medusa ===")
        print(f"model={args.medusa_model}")
        print(f"max_ctx={medusa_max_ctx}")
        print(f"kv_cache_len={res_m['kv_cache_len']}")
        print(f"total_tokens={res_m['total_tokens']}  total_time={res_m['total_time_s']:.2f}s  tps={res_m['tokens_per_second']:.2f}")

        del medusa_model
        cleanup_jittor()
        cleanup_torch()

    elif args.run == "jt_baseline":
        if args.jedusa_path and args.jedusa_path not in sys.path:
            sys.path.append(args.jedusa_path)

        jt = _try_import_jittor()
        if jt is None:
            raise RuntimeError("Requested jt_baseline run, but jittor import failed.")
        try:
            jt.flags.use_cuda = 1
        except Exception:
            pass

        MedusaModel = resolve_medusa_class(args.medusa_impl)

        dprint(args.debug_medusa, f"[jt-baseline] {now_ts()} loading jittor vicuna: {args.medusa_model}")
        if hasattr(MedusaModel, "from_pretrained") and callable(getattr(MedusaModel, "from_pretrained")):
            sig = inspect.signature(MedusaModel.from_pretrained)
            kw = {}
            for k, v in {
                # heads are not used here, but keep the structure valid
                "medusa_num_heads": 1,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }.items():
                if k in sig.parameters:
                    kw[k] = v
            jt_wrapper = MedusaModel.from_pretrained(args.medusa_model, **kw)
        else:
            jt_wrapper = MedusaModel(args.medusa_model)

        if hasattr(jt_wrapper, "eval"):
            jt_wrapper.eval()

        # Use pure backbone for decoding
        jt_model = getattr(jt_wrapper, "base_model", jt_wrapper)

        jt_max_ctx = infer_max_ctx(jt_model, tok, args.max_ctx)
        cap_ctx = None
        try:
            if hasattr(jt_model, "config") and getattr(jt_model.config, "max_position_embeddings", None):
                cap_ctx = int(jt_model.config.max_position_embeddings)
        except Exception:
            cap_ctx = None
        if cap_ctx is not None:
            jt_max_ctx = int(min(int(jt_max_ctx), int(cap_ctx)))

        dprint(args.debug_medusa, f"[jt-baseline] inferred_ctx={jt_max_ctx} cap_ctx={cap_ctx}")

        def jt_turn12(prompt1, stop_strs1, stop_token_ids1, warmup_only=False, turns=None):
            a1, n1 = gen_jt_baseline(
                jt_model, tok, prompt1, stop_strs1, stop_token_ids1 or [],
                jt_max_ctx, args.max_new_tokens, args.temperature, args.top_p,
                debug=args.debug_medusa,
            )
            if warmup_only:
                return a1, "", n1, 0

            prompt2, conv2 = build_prompt(args.conv, [turns[0], turns[1]], [a1])
            stop_strs2, stop_token_ids2 = get_conv_stop(conv2)
            a2, n2 = gen_jt_baseline(
                jt_model, tok, prompt2, stop_strs2, stop_token_ids2 or [],
                jt_max_ctx, args.max_new_tokens, args.temperature, args.top_p,
                debug=args.debug_medusa,
            )
            return a1, a2, n1, n2

        res_jt = run_benchmark(
            mode="jt_baseline",
            qs=qs,
            tok=tok,
            conv_name=args.conv,
            warmup=args.warmup,
            gen_turn12_fn=jt_turn12,
            time_fn=time_jittor,
            answer_file=args.jt_answer_file,
            model_id=args.jt_model_id,
            detail_jsonl=args.detail_jsonl,
        )
        res_jt["model"] = args.medusa_model
        res_jt["max_ctx"] = int(jt_max_ctx)
        combined["results"]["jt_baseline"] = res_jt

        print("\n=== jt_baseline ===")
        print(f"model={args.medusa_model}")
        print(f"max_ctx={jt_max_ctx}")
        print(f"total_tokens={res_jt['total_tokens']}  total_time={res_jt['total_time_s']:.2f}s  tps={res_jt['tokens_per_second']:.2f}")

        del jt_wrapper
        cleanup_jittor()
        cleanup_torch()

    elif args.run == "baseline":
        baseline_model = AutoModelForCausalLM.from_pretrained(
            args.baseline_model, torch_dtype=torch.float16, device_map="auto"
        )
        baseline_model.eval()
        baseline_max_ctx = infer_max_ctx(baseline_model, tok, args.max_ctx)

        def baseline_turn12(prompt1, stop_strs1, stop_token_ids1, warmup_only=False, turns=None):
            a1, n1 = gen_baseline(
                baseline_model, tok, prompt1, stop_strs1, stop_token_ids1 or [],
                baseline_max_ctx, args.max_new_tokens, args.temperature, args.top_p
            )
            if warmup_only:
                return a1, "", n1, 0

            prompt2, conv2 = build_prompt(args.conv, [turns[0], turns[1]], [a1])
            stop_strs2, stop_token_ids2 = get_conv_stop(conv2)
            a2, n2 = gen_baseline(
                baseline_model, tok, prompt2, stop_strs2, stop_token_ids2,
                baseline_max_ctx, args.max_new_tokens, args.temperature, args.top_p
            )
            return a1, a2, n1, n2

        res_b = run_benchmark(
            mode="baseline",
            qs=qs,
            tok=tok,
            conv_name=args.conv,
            warmup=args.warmup,
            gen_turn12_fn=baseline_turn12,
            time_fn=time_torch,
            answer_file=args.baseline_answer_file,
            model_id=args.baseline_model_id,
            detail_jsonl=args.detail_jsonl,
        )
        res_b["model"] = args.baseline_model
        res_b["max_ctx"] = int(baseline_max_ctx)
        combined["results"]["baseline"] = res_b

        print("\n=== baseline ===")
        print(f"model={args.baseline_model}")
        print(f"max_ctx={baseline_max_ctx}")
        print(f"total_tokens={res_b['total_tokens']}  total_time={res_b['total_time_s']:.2f}s  tps={res_b['tokens_per_second']:.2f}")

        del baseline_model
        cleanup_torch()

    if args.out_json:
        save_json(combined, args.out_json)
        print(f"\n✅ wrote json to: {args.out_json}", flush=True)

    if log_handle is not None:
        try:
            log_handle.write(f"\n===== {args.run} END {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log_handle.flush()
            log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()