# import argparse, json, time, os
# from collections import defaultdict
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Tuple
# import shortuuid
# import torch
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# # 确保导入 get_conv_stop
# from fastchat.conversation import get_conv_template  # 添加这一行

# def _to_torch_on_device(x, device):
#     if hasattr(x, "numpy"):
#         x = x.numpy()
#     if isinstance(x, np.ndarray):
#         x = torch.from_numpy(x)
#     if torch.is_tensor(x):
#         return x.to(device)
#     return torch.tensor(x, device=device)
# def get_conv_stop(conv) -> Tuple[List[str], List[int]]:
#     """
#     Extract stop strings / stop token ids from FastChat template (best-effort).
#     Different versions may have slightly different fields.
#     """
#     stop_strs = []
#     stop_token_ids = []

#     if hasattr(conv, "stop_str") and conv.stop_str:
#         if isinstance(conv.stop_str, str):
#             stop_strs.append(conv.stop_str)
#         else:
#             stop_strs.extend(list(conv.stop_str))

#     if hasattr(conv, "stop_token_ids") and conv.stop_token_ids:
#         stop_token_ids.extend(list(conv.stop_token_ids))

#     # Some templates rely on sep / sep2 as implicit stops
#     # (we only add them if they look like real separators)
#     for name in ["sep", "sep2"]:
#         if hasattr(conv, name):
#             s = getattr(conv, name)
#             if isinstance(s, str) and len(s) > 0 and s not in stop_strs:
#                 # Avoid overly common whitespace-like separators
#                 if s.strip() != "":
#                     stop_strs.append(s)

#     return stop_strs, stop_token_ids
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
# # Timing helpers
# # ----------------------------
# def cuda_time(fn):
#     torch.cuda.synchronize()
#     t0 = time.time()
#     out = fn()
#     torch.cuda.synchronize()
#     t1 = time.time()
#     return out, (t1 - t0)

# # ----------------------------
# # Stop criteria
# # ----------------------------
# class StopOnSubstrings(StoppingCriteria):
#     def __init__(self, tokenizer, stop_strs: List[str], start_len: int, window: int = 64):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.stop_strs = [s for s in stop_strs if s]
#         self.start_len = start_len
#         self.window = window

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         if not self.stop_strs:
#             return False
#         gen = input_ids[0, self.start_len:]
#         if gen.numel() <= 0:
#             return False
#         tail = gen[-self.window:].tolist()
#         text = self.tokenizer.decode(tail, skip_special_tokens=False)
#         return any(s in text for s in self.stop_strs)

# def cut_by_stop(text: str, stop_strs: List[str]) -> str:
#     if not stop_strs:
#         return text
#     idxs = [text.find(s) for s in stop_strs if s and text.find(s) != -1]
#     if not idxs:
#         return text
#     cut = min(idxs)
#     return text[:cut]

# # ----------------------------
# # Prompt building
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
# # Input truncation
# # ----------------------------
# def infer_max_ctx(model, tokenizer, override: Optional[int]) -> int:
#     if override is not None:
#         return int(override)

#     if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
#         return int(model.config.max_position_embeddings)

#     m = getattr(tokenizer, "model_max_length", None)
#     if m is not None and isinstance(m, int) and m < 10**6:
#         return int(m)

#     return 2048

# def tokenize_prompt(
#     tok,
#     prompt: str,
#     device: torch.device,
#     max_input_len: int,
# ) -> Dict[str, torch.Tensor]:
#     enc = tok(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max(1, int(max_input_len)),
#     )
#     return {k: v.to(device) for k, v in enc.items()}

# # ----------------------------
# # Generation: baseline
# # ----------------------------
# def gen_baseline(
#     model,
#     tok,
#     prompt: str,
#     conv_stop_strs: List[str],
#     conv_stop_token_ids: List[int],
#     max_ctx: int,
#     max_new_tokens: int,
#     temperature: float,
#     top_p: float,
# ) -> Tuple[str, int]:
#     safety = 16
#     max_in = max(1, max_ctx - max_new_tokens - safety)
#     inputs = tokenize_prompt(tok, prompt, model.device, max_in)

#     do_sample = temperature > 0
#     eos_ids = []
#     if model.config.eos_token_id is not None:
#         eos_ids.append(int(model.config.eos_token_id))
#     eos_ids.extend([int(x) for x in conv_stop_token_ids if x is not None])

#     start_len = inputs["input_ids"].shape[1]
#     stop_criteria = StoppingCriteriaList()
#     if conv_stop_strs:
#         stop_criteria.append(StopOnSubstrings(tok, conv_stop_strs, start_len=start_len, window=64))

#     out_ids = model.generate(
#         **inputs,
#         do_sample=do_sample,
#         temperature=temperature if do_sample else None,
#         top_p=top_p if do_sample else None,
#         max_new_tokens=max_new_tokens,
#         use_cache=True,
#         eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
#         pad_token_id=model.config.eos_token_id,
#         stopping_criteria=stop_criteria if len(stop_criteria) > 0 else None,
#     )

#     gen_ids = out_ids[0, start_len:]
#     text = tok.decode(gen_ids, skip_special_tokens=True)
#     text = cut_by_stop(text, conv_stop_strs).strip()

#     n_out = count_output_tokens(tok, text)
#     return text, n_out

# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--questions", type=str, required=True)
#     ap.add_argument("--mode", choices=["baseline"], required=True)
#     ap.add_argument("--model", type=str, required=True, help="模型路径")
#     ap.add_argument("--conv", type=str, default="vicuna_v1.1", help="FastChat conv 模板名")

#     ap.add_argument("--max_new_tokens", type=int, default=256)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_p", type=float, default=0.95)

#     ap.add_argument("--limit", type=int, default=None)
#     ap.add_argument("--start", type=int, default=0)
#     ap.add_argument("--warmup", type=int, default=3)

#     ap.add_argument("--max_ctx", type=int, default=None, help="Override model max context length")
#     ap.add_argument("--out_json", type=str, default=None, help="输出 JSON 结果")

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

#     torch.set_grad_enabled(False)

#     tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         torch_dtype=torch.float16,
#         device_map="auto",
#     )
#     model.eval()
#     max_ctx = infer_max_ctx(model, tok, args.max_ctx)

#     def gen_fn(prompt, conv_stop_strs, conv_stop_token_ids):
#         return gen_baseline(
#             model=model,
#             tok=tok,
#             prompt=prompt,
#             conv_stop_strs=conv_stop_strs,
#             conv_stop_token_ids=conv_stop_token_ids,
#             max_ctx=max_ctx,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_p=args.top_p,
#         )

#     # Warmup
#     for i in range(min(args.warmup, len(qs))):
#         q = qs[i]
#         prompt, conv = build_prompt(args.conv, [q["turns"][0]], [])
#         stop_strs, stop_token_ids = get_conv_stop(conv)
#         _ = gen_fn(prompt, stop_strs, stop_token_ids)

#     total_time = 0.0
#     total_tokens = 0

#     cat_time = defaultdict(float)
#     cat_tokens = defaultdict(int)
#     cat_cnt = defaultdict(int)

#     pbar = tqdm(qs, desc="baseline", unit="q")
#     for q in pbar:
#         cat = q.get("category", "unknown")
#         turns = q["turns"]

#         # turn1
#         prompt1, conv1 = build_prompt(args.conv, [turns[0]], [])
#         stop_strs1, stop_token_ids1 = get_conv_stop(conv1)
#         (a1, n1), dt1 = cuda_time(lambda: gen_fn(prompt1, stop_strs1, stop_token_ids1))

#         # turn2
#         prompt2, conv2 = build_prompt(args.conv, [turns[0], turns[1]], [a1])
#         stop_strs2, stop_token_ids2 = get_conv_stop(conv2)
#         (a2, n2), dt2 = cuda_time(lambda: gen_fn(prompt2, stop_strs2, stop_token_ids2))

#         dt = dt1 + dt2
#         nt = n1 + n2

#         total_time += dt
#         total_tokens += nt

#         cat_time[cat] += dt
#         cat_tokens[cat] += nt
#         cat_cnt[cat] += 1

#         tps = total_tokens / max(total_time, 1e-9)
#         pbar.set_postfix(tps=f"{tps:.2f}", dt=f"{dt:.2f}s", cat=cat)

#         if args.out_json:
#             append_answer(args.out_json, q["question_id"], args.model, [a1, a2])

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
#         "mode": "baseline",
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
#     }

#     print(f"mode=baseline")
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
import argparse, json, time, os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shortuuid
import jittor as jt  # 使用 Jittor 替代 PyTorch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer  # 使用 transformers 的 tokenizer（仍然需要）

# 确保导入 get_conv_stop
from fastchat.conversation import get_conv_template

def _to_jittor_on_device(x, device):
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = jt.array(x)
    if isinstance(x, jt.Var):
        return x
    return jt.array(x, device=device)

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
    jt.sync()
    t0 = time.time()
    out = fn()
    jt.sync()
    t1 = time.time()
    return out, (t1 - t0)

# ----------------------------
# Stop criteria
# ----------------------------
class StopOnSubstrings:
    def __init__(self, tokenizer, stop_strs: List[str], start_len: int, window: int = 64):
        self.tokenizer = tokenizer
        self.stop_strs = [s for s in stop_strs if s]
        self.start_len = start_len
        self.window = window

    def __call__(self, input_ids: jt.Var, scores: jt.Var, **kwargs) -> bool:
        if not self.stop_strs:
            return False
        gen = input_ids[0, self.start_len:]
        if gen.numel() <= 0:
            return False
        tail = gen[-self.window:].tolist()
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return any(s in text for s in self.stop_strs)

def cut_by_stop(text: str, stop_strs: List[str]) -> str:
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
    return len(tok.encode(text, add_special_tokens=False))

# ----------------------------
# Input truncation
# ----------------------------
def infer_max_ctx(model, tokenizer, override: Optional[int]) -> int:
    if override is not None:
        return int(override)

    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
        return int(model.config.max_position_embeddings)

    m = getattr(tokenizer, "model_max_length", None)
    if m is not None and isinstance(m, int) and m < 10**6:
        return int(m)

    return 2048

def tokenize_prompt(
    tok,
    prompt: str,
    device: jt.Var,
    max_input_len: int,
) -> Dict[str, jt.Var]:
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, int(max_input_len)),
    )
    return {k: _to_jittor_on_device(v, device) for k, v in enc.items()}

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
    safety = 16
    max_in = max(1, max_ctx - max_new_tokens - safety)
    inputs = tokenize_prompt(tok, prompt, model.device, max_in)

    do_sample = temperature > 0
    eos_ids = []
    if model.config.eos_token_id is not None:
        eos_ids.append(int(model.config.eos_token_id))
    eos_ids.extend([int(x) for x in conv_stop_token_ids if x is not None])

    start_len = inputs["input_ids"].shape[1]
    stop_criteria = []
    if conv_stop_strs:
        stop_criteria.append(StopOnSubstrings(tok, conv_stop_strs, start_len=start_len, window=64))

    # Jittor 推理部分需要自己实现，这里是一个示意
    # 通过 Jittor 对模型进行推理

    out_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
        pad_token_id=model.config.eos_token_id,
        stopping_criteria=stop_criteria if len(stop_criteria) > 0 else None,
    )

    gen_ids = out_ids[0, start_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    text = cut_by_stop(text, conv_stop_strs).strip()

    n_out = count_output_tokens(tok, text)
    return text, n_out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=str, required=True)
    ap.add_argument("--mode", choices=["baseline"], required=True)
    ap.add_argument("--model", type=str, required=True, help="模型路径")
    ap.add_argument("--conv", type=str, default="vicuna_v1.1", help="FastChat conv 模板名")

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)

    ap.add_argument("--max_ctx", type=int, default=None, help="Override model max context length")
    ap.add_argument("--out_json", type=str, default=None, help="输出 JSON 结果")

    args = ap.parse_args()

    # 禁用 GPU 计算（禁用梯度）
    jt.flags.use_cuda = False  # 禁用 GPU 计算

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

    # 使用 Jittor 加载模型（需要自定义加载部分）
    # model = jt.load(args.model)  # 示例，需根据实际情况进行调整
    model = ...  # 使用适当的方式加载 Jittor 模型

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # 定义 gen_fn 函数
    def gen_fn(prompt, conv_stop_strs, conv_stop_token_ids):
        return gen_baseline(
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

    # Warmup
    for i in range(min(args.warmup, len(qs))):
        q = qs[i]
        prompt, conv = build_prompt(args.conv, [q["turns"][0]], [])
        stop_strs, stop_token_ids = get_conv_stop(conv)
        _ = gen_fn(prompt, stop_strs, stop_token_ids)

    total_time = 0.0
    total_tokens = 0

    cat_time = defaultdict(float)
    cat_tokens = defaultdict(int)
    cat_cnt = defaultdict(int)

    pbar = tqdm(qs, desc="baseline", unit="q")
    for q in pbar:
        cat = q.get("category", "unknown")
        turns = q["turns"]

        # turn1
        prompt1, conv1 = build_prompt(args.conv, [turns[0]], [])
        stop_strs1, stop_token_ids1 = get_conv_stop(conv1)
        (a1, n1), dt1 = cuda_time(lambda: gen_fn(prompt1, stop_strs1, stop_token_ids1))

        # turn2
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

        if args.out_json:
            append_answer(args.out_json, q["question_id"], args.model, [a1, a2])

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
        "mode": "baseline",
        "model": args.model,
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

    print(f"mode=baseline")
    print(f"model={args.model}")
    print(f"conv={args.conv}  max_ctx={max_ctx}")
    print(f"questions={len(qs)}  max_new_tokens={args.max_new_tokens}  temp={args.temperature}  top_p={args.top_p}")
    print(f"total_tokens={total_tokens}  total_time={total_time:.2f}s")
    print(f"tokens_per_second={overall_tps:.2f}")

    for cat in sorted(per_cat.keys()):
        print(f"[{cat:12s}] tps={per_cat[cat]['tokens_per_second']:.2f}  "
              f"tokens={per_cat[cat]['tokens']}  time={per_cat[cat]['time_s']:.2f}s  n={per_cat[cat]['n_questions']}")

    if args.out_json:
        save_json(result, args.out_json)

if __name__ == "__main__":
    main()
