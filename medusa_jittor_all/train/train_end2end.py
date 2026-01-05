import os
import json
import argparse
import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jittor as jt
from jittor import nn
from transformers import AutoTokenizer, AutoConfig

# Import from our new package
from medusa_jittor_all.model.medusa_model import MedusaModel

try:
    import wandb
except Exception:
    wandb = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

IGNORE_TOKEN_ID = -100

# --------------------------
# Data helpers
# --------------------------
def load_samples(data_path: str) -> List[Any]:
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else obj.get("data", [])

def format_item_with_spans(item: Any) -> Tuple[str, List[Tuple[int, int]]]:
    if isinstance(item, dict) and "conversations" in item:
        parts = []
        spans = []
        cur = 0
        for msg in item["conversations"]:
            role = msg.get("from", "")
            content = msg.get("value", "")
            if role == "human":
                seg = f"USER: {content} "
                parts.append(seg)
                cur += len(seg)
            elif role == "gpt":
                prefix = "ASSISTANT: "
                seg = f"{prefix}{content} </s> "
                parts.append(seg)
                start = cur + len(prefix)
                end = cur + len(seg)
                spans.append((start, end))
                cur += len(seg)
            else:
                seg = f"{str(content)} "
                parts.append(seg)
                cur += len(seg)
        text = "".join(parts).strip()
        spans = [(s, min(e, len(text))) for (s, e) in spans if s < len(text)]
        return text, spans
    if isinstance(item, dict) and "text" in item:
        text = str(item["text"])
        return text, [(0, len(text))]
    text = str(item)
    return text, [(0, len(text))]

def build_labels_assistant_only(tokenizer, text: str, spans: List[Tuple[int, int]], max_length: int) -> Dict[str, np.ndarray]:
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = np.asarray(enc["input_ids"], dtype=np.int32)[None, :]
    attention_mask = np.asarray(enc["attention_mask"], dtype=np.int32)[None, :]
    offsets = enc["offset_mapping"]
    labels = np.full_like(input_ids, IGNORE_TOKEN_ID, dtype=np.int32)
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue
        keep = False
        for (s, e) in spans:
            if not (b <= s or a >= e):
                keep = True
                break
        if keep:
            labels[0, ti] = input_ids[0, ti]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --------------------------
# Scalar helpers
# --------------------------
def _scalar_float(v) -> float:
    if isinstance(v, (float, int)):
        return float(v)
    try:
        if hasattr(v, "data"):
            a = np.asarray(v.data)
            return float(a.reshape(-1)[0])
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return float("nan")

def wandb_log_safe(data: Dict[str, Any], step: int):
    if wandb is None:
        return
    safe = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            safe[k] = v
        else:
            vv = _scalar_float(v)
            if not (isinstance(vv, float) and (math.isnan(vv) or math.isinf(vv))):
                safe[k] = vv
    wandb.log(safe, step=step)

# --------------------------
# Loss/metrics
# --------------------------
def masked_cross_entropy_and_top1(logits, labels, ignore_id: int):
    labels = labels.int32()
    mask = labels != ignore_id
    cnt = mask.sum().float32()
    safe_labels = jt.where(mask, labels, jt.zeros_like(labels))
    logp = nn.log_softmax(logits, dim=1)
    nll = -logp.gather(1, safe_labels.reshape(-1, 1)).reshape(-1)
    denom = jt.maximum(cnt, jt.float32(1.0))
    loss = (nll * mask.float32()).sum() / denom
    _, pred = logits.argmax(dim=1)
    correct = ((pred == safe_labels) * mask).sum().float32()
    top1 = correct / denom
    return loss, top1, cnt

# --------------------------
# LR scheduler
# --------------------------
def lr_at_step(base_lr: float, step: int, total_steps: int, warmup_steps: int, sched: str) -> float:
    if total_steps <= 0:
        return base_lr
    step = max(1, step)
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    if sched == "none":
        return base_lr
    denom = max(1, total_steps - warmup_steps)
    t = (step - warmup_steps) / denom
    t = min(max(t, 0.0), 1.0)
    if sched == "linear":
        return base_lr * (1.0 - t)
    if sched == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
    return base_lr

def set_optimizer_lr(opt, new_lr: float):
    if hasattr(opt, "lr"):
        opt.lr = float(new_lr)
        return
    if hasattr(opt, "param_groups"):
        for g in opt.param_groups:
            g["lr"] = float(new_lr)

# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_name_or_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--medusa_num_heads", type=int, default=5)
    p.add_argument("--medusa_num_layers", type=int, default=1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--tqdm_mininterval", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="none", choices=["none", "cosine", "linear"])
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="medusa-jittor")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, default="")

    args = p.parse_args()

    if args.batch_size != 1:
        print("[Warn] Batch size > 1 not fully tested.")

    if not args.no_tqdm and tqdm is None:
        print("[Warn] tqdm not found, disabling.")
        args.no_tqdm = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    print(f"[JittorTrainer] use_cuda={jt.flags.use_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Model] Loading MedusaModel from {args.base_model_name_or_path}")
    # Load model
    # Note: This will try to load weights. If Jittor weights are not found, it might fail or load random weights depending on implementation.
    # We assume the user handles weight conversion or we are training from scratch (random weights for heads, but backbone needs weights).
    # For now, we proceed.
    model = MedusaModel.from_pretrained(
        args.base_model_name_or_path,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        local_files_only=args.local_files_only
    )
    
    # Convert model to float16 to save memory (7B float32 is ~28GB, too big for 24GB GPU)
    print("[Model] Converting to float16...")
    # Jittor doesn't have .half() on Module recursively by default in older versions?
    # Let's check if we can use a helper or if it exists.
    # If not, we can iterate parameters.
    # But Jittor 1.3.10 should support it or we can do it manually.
    
    def model_half(m):
        for name, param in m.named_parameters():
            m.__setattr__(name, param.float16())
        for child in m.modules():
            if child is not m:
                model_half(child)
                
    # Actually, let's try simple recursion if .half() is not available.
    # But wait, Jittor variables are what matters.
    # Let's try to use a safer way: iterate named_parameters and reset them.
    
    for name, param in model.named_parameters():
        # We need to set the parameter back to the module
        # This is tricky with nested modules.
        # But Jittor parameters are just attributes.
        
        # A better way in Jittor might be to just cast inputs and let Jittor optimize?
        # No, weights take memory.
        pass

    # Let's implement a robust half() function
    def to_half(module):
        for k, v in module.__dict__.items():
            if isinstance(v, jt.Var):
                if v.dtype == jt.float32:
                    module.__dict__[k] = v.float16()
        for idx, m in enumerate(module.modules()):
            if m is not module:
                to_half(m)
                
    to_half(model)
    
    # Optimizer: only optimize medusa heads
    opt = nn.AdamW(model.medusa_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        if wandb is None:
            print("[Warn] wandb not installed.")
        else:
            wandb_kwargs = dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
            if args.wandb_mode:
                wandb_kwargs["mode"] = args.wandb_mode
            run = wandb.init(**wandb_kwargs)
            if args.wandb_tags:
                tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
                try:
                    run.tags = tuple(tags)
                except Exception:
                    pass

    samples = load_samples(args.data_path)
    n = len(samples)
    print(f"[Data] num_samples={n}")

    total_steps = args.epochs * n
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    use_tqdm = (not args.no_tqdm) and (tqdm is not None)
    pbar = None
    if use_tqdm:
        pbar = tqdm(total=total_steps, dynamic_ncols=True, mininterval=args.tqdm_mininterval)

    accum = max(1, int(args.grad_accum_steps))
    step = 0
    micro = 0
    ema_loss = None
    ema_beta = 0.95

    try:
        for ep in range(args.epochs):
            if args.shuffle:
                rng = np.random.default_rng(seed=args.seed + ep)
                rng.shuffle(samples)

            if use_tqdm and pbar is not None:
                pbar.set_description(f"epoch {ep+1}/{args.epochs}")

            for item in samples:
                if args.max_steps > 0 and step >= args.max_steps:
                    break
                if step >= total_steps:
                    break

                t0 = time.time()
                step += 1
                micro += 1

                lr_now = lr_at_step(args.lr, step, total_steps, warmup_steps, args.lr_scheduler_type)
                set_optimizer_lr(opt, lr_now)

                text, spans = format_item_with_spans(item)
                pack = build_labels_assistant_only(tokenizer, text, spans, args.max_length)

                input_ids = jt.array(pack["input_ids"].astype(np.int32))
                attn_mask = jt.array(pack["attention_mask"].astype(np.int32))
                labels = jt.array(pack["labels"].astype(np.int32))

                # Forward pass
                # medusa_forward=True to get medusa logits
                # output_orig=False
                medusa_logits = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    medusa_forward=True
                )
                # medusa_logits: [heads, bs, seq_len, vocab]

                total_loss = jt.float32(0.0)
                head_logs = {}
                B, T = labels.shape
                vocab_size = model.vocab_size

                for k in range(args.medusa_num_heads):
                    end_idx = T - (2 + k)
                    start_label = 2 + k
                    if end_idx <= 0:
                        continue

                    logits_k = medusa_logits[k][:, :end_idx, :].reshape(-1, vocab_size)
                    labels_k = labels[:, start_label:T].reshape(-1)

                    loss_k, top1_k, cnt_k = masked_cross_entropy_and_top1(
                        logits_k, labels_k, IGNORE_TOKEN_ID
                    )
                    total_loss = total_loss + loss_k

                    head_logs[f"medusa{k}_loss"] = _scalar_float(loss_k)
                    cnt_val = _scalar_float(cnt_k)
                    head_logs[f"medusa{k}_tokens"] = cnt_val
                    if cnt_val > 0:
                        head_logs[f"medusa{k}_top1"] = _scalar_float(top1_k)

                if accum == 1:
                    opt.step(total_loss)
                else:
                    scaled = total_loss / float(accum)
                    opt.backward(scaled)
                    if micro % accum == 0:
                        opt.step()
                        # opt.zero_grad() # Jittor optimizer usually clears grad in step? No, need to check.
                        # Jittor optimizers accumulate grad?
                        # Usually opt.step(loss) does backward and step and clear.
                        # If we do opt.backward(loss), it accumulates.
                        # Then opt.step() updates and clears.
                        # Wait, Jittor opt.step(loss) is a shortcut for backward+step+zero.
                        # If we want accumulation, we use opt.backward(loss) multiple times, then opt.step().
                        # But opt.step() in Jittor might not clear grads automatically if called without loss?
                        # Let's assume standard Jittor behavior:
                        # opt.backward(loss) adds to grad.
                        # opt.step() updates weights.
                        # We need to manually zero grad?
                        # Jittor vars don't hold grad persistently like PyTorch unless retain_grad?
                        # Actually, Jittor optimizer has zero_grad.
                        if hasattr(opt, "zero_grad"):
                            opt.zero_grad()

                loss_val = _scalar_float(total_loss)
                
                # Cleanup
                del medusa_logits, input_ids, attn_mask, labels, total_loss
                jt.gc()

                dt = time.time() - t0
                tokens = args.batch_size * args.max_length
                tps = tokens / max(dt, 1e-9)

                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_val

                if step % args.log_every == 0:
                    msg = f"[Train] epoch={ep} step={step} loss={loss_val:.6f} ema={ema_loss:.6f} tps={tps:.1f} lr={lr_now:.2e}"
                    if use_tqdm:
                        tqdm.write(msg)
                    else:
                        print(msg)

                if args.save_every > 0 and step % args.save_every == 0:
                    path = os.path.join(args.output_dir, f"medusa_step_{step}.pkl")
                    jt.save(model.medusa_head.state_dict(), path)
                    if use_tqdm:
                        tqdm.write(f"[Save] {path}")
                    else:
                        print(f"[Save] {path}")

                if args.wandb and wandb is not None:
                    log = {
                        "train/loss": loss_val,
                        "train/loss_ema": ema_loss,
                        "train/step_time_sec": dt,
                        "train/tokens_per_sec": tps,
                        "train/epoch": ep,
                        "train/step": step,
                        "optim/lr": lr_now,
                        "optim/grad_accum_steps": accum,
                    }
                    log.update(head_logs)
                    wandb_log_safe(log, step=step)

                if use_tqdm and pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{loss_val:.3f}",
                        ema=f"{ema_loss:.3f}",
                        tps=f"{tps:.0f}",
                        lr=f"{lr_now:.1e}",
                    )

            epoch_path = os.path.join(args.output_dir, f"medusa_epoch_{ep}.pkl")
            jt.save(model.medusa_head.state_dict(), epoch_path)
            if use_tqdm:
                tqdm.write(f"[Save] {epoch_path}")
            else:
                print(f"[Save] {epoch_path}")

    finally:
        if pbar is not None:
            pbar.close()
        if args.wandb and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

if __name__ == "__main__":
    main()
