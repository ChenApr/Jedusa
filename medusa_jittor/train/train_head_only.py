# medusa_jittor/train/train_head_only.py
import os
import json
import argparse
import socket
import struct
import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import jittor as jt
from jittor import nn
from transformers import AutoTokenizer, AutoConfig

from medusa_jittor.model.medusa_head import MedusaHead

try:
    import wandb
except Exception:
    wandb = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

IGNORE_TOKEN_ID = -100  # match HF LabelSmoother.ignore_index


# --------------------------
# Socket protocol: send ids/mask -> recv hidden
# --------------------------
def send_batch(sock, input_ids: np.ndarray, attention_mask: np.ndarray, meta: dict):
    header = {
        "meta": meta,
        "input_ids": {"dtype": str(input_ids.dtype), "shape": list(input_ids.shape)},
        "attention_mask": {"dtype": str(attention_mask.dtype), "shape": list(attention_mask.shape)},
    }
    blobs = []
    offset = 0
    for name, arr in [("input_ids", input_ids), ("attention_mask", attention_mask)]:
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        header[name]["offset"] = offset
        header[name]["nbytes"] = len(b)
        blobs.append(b)
        offset += len(b)

    header_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack("!I", len(header_bytes)) + header_bytes + b"".join(blobs)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_hidden(sock) -> np.ndarray:
    n = struct.unpack("!I", recv_exact(sock, 4))[0]
    payload = recv_exact(sock, n)
    header_len = struct.unpack("!I", payload[:4])[0]
    header = json.loads(payload[4: 4 + header_len].decode("utf-8"))
    blob = memoryview(payload[4 + header_len:])

    info = header["hidden"]
    offset, nbytes = info["offset"], info["nbytes"]
    hidden = np.frombuffer(blob[offset: offset + nbytes], dtype=info["dtype"]).reshape(info["shape"])
    return hidden


# --------------------------
# Data helpers
# --------------------------
def load_samples(data_path: str) -> List[Any]:
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else obj.get("data", [])


def format_item_with_spans(item: Any) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Return:
      text: formatted training string
      assistant_spans: list of (start_char, end_char) ranges where labels should be computed
    Only compute labels on assistant content (roughly aligns with HF scripts).
    """
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
                end = cur + len(seg)  # include </s> for training
                spans.append((start, end))

                cur += len(seg)
            else:
                seg = f"{str(content)} "
                parts.append(seg)
                cur += len(seg)

        text = "".join(parts).strip()
        # strip() may remove trailing spaces, clamp spans
        spans = [(s, min(e, len(text))) for (s, e) in spans if s < len(text)]
        return text, spans

    if isinstance(item, dict) and "text" in item:
        text = str(item["text"])
        return text, [(0, len(text))]

    text = str(item)
    return text, [(0, len(text))]


def build_labels_assistant_only(tokenizer, text: str, spans: List[Tuple[int, int]], max_length: int) -> Dict[str, np.ndarray]:
    """
    Tokenize with offsets, and create labels where tokens overlapping assistant spans keep their ids,
    otherwise IGNORE_TOKEN_ID.
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = np.asarray(enc["input_ids"], dtype=np.int32)[None, :]            # [1,T]
    attention_mask = np.asarray(enc["attention_mask"], dtype=np.int32)[None, :]  # [1,T]
    offsets = enc["offset_mapping"]  # list[(start,end)] length T

    labels = np.full_like(input_ids, IGNORE_TOKEN_ID, dtype=np.int32)  # [1,T]
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue
        keep = False
        for (s, e) in spans:
            if not (b <= s or a >= e):  # overlap
                keep = True
                break
        if keep:
            labels[0, ti] = input_ids[0, ti]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --------------------------
# Scalar helpers (VERY IMPORTANT for wandb + python if)
# --------------------------
def _scalar_float(v) -> float:
    """
    jt.Var / numpy scalar / python number -> python float
    """
    if isinstance(v, (float, int)):
        return float(v)
    try:
        if hasattr(v, "data"):  # jt.Var
            a = np.asarray(v.data)
            return float(a.reshape(-1)[0])
        if hasattr(v, "item"):  # numpy scalar
            return float(v.item())
        return float(v)
    except Exception:
        return float("nan")


def wandb_log_safe(data: Dict[str, Any], step: int):
    """
    Ensure all values are JSON-serializable (python scalars only).
    """
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
# Loss/metrics (ignore_index safe)
# --------------------------
def masked_cross_entropy_and_top1(logits, labels, ignore_id: int):
    """
    logits: [N, V] jt.Var
    labels: [N] jt.Var (int)
    return: loss (scalar Var), top1 (scalar Var), cnt (scalar Var)
    """
    labels = labels.int32()
    mask = labels != ignore_id
    cnt = mask.sum().float32()

    safe_labels = jt.where(mask, labels, jt.zeros_like(labels))

    logp = nn.log_softmax(logits, dim=1)  # [N,V]
    nll = -logp.gather(1, safe_labels.reshape(-1, 1)).reshape(-1)  # [N]

    denom = jt.maximum(cnt, jt.float32(1.0))
    loss = (nll * mask.float32()).sum() / denom

    # jittor argmax returns (max_vals, argmax_idx)
    _, pred = logits.argmax(dim=1)  # pred: [N]
    correct = ((pred == safe_labels) * mask).sum().float32()
    top1 = correct / denom
    return loss, top1, cnt


# --------------------------
# LR scheduler (simple warmup+cosine/linear)
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
    p.add_argument("--worker_host", type=str, default="127.0.0.1")
    p.add_argument("--worker_port", type=int, default=5000)

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

    # tqdm
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--tqdm_mininterval", type=float, default=1.0)

    # compare-aligned training extras
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="none", choices=["none", "cosine", "linear"])

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="medusa-jittor")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, default="")

    args = p.parse_args()

    if args.batch_size != 1:
        print("[Warn] 当前实现建议 batch_size=1（否则要扩展 socket 协议 + padding 拼 batch）")

    if not args.no_tqdm and tqdm is None:
        raise RuntimeError("tqdm not installed. Run: pip install tqdm  或者加 --no_tqdm 关闭进度条")

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # jittor device
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    print(f"[JittorTrainer] use_cuda={jt.flags.use_cuda}")

    # tokenizer + config
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(args.base_model_name_or_path, local_files_only=args.local_files_only)
    hidden_size = int(cfg.hidden_size)
    vocab_size = int(cfg.vocab_size)
    print(f"[BackboneCfg] hidden={hidden_size} vocab={vocab_size} max_length={args.max_length}")

    # model + optimizer
    medusa = MedusaHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
    )
    opt = nn.AdamW(medusa.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)

    # wandb init
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed. Run: pip install wandb")
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

    # connect worker
    sock = socket.create_connection((args.worker_host, args.worker_port))
    print(f"[JittorTrainer] connected to worker {args.worker_host}:{args.worker_port}")

    samples = load_samples(args.data_path)
    n = len(samples)
    print(f"[Data] num_samples={n}")

    # steps
    total_steps = args.epochs * n
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    # tqdm (single bar)
    use_tqdm = (not args.no_tqdm) and (tqdm is not None)
    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=total_steps,
            dynamic_ncols=True,
            mininterval=args.tqdm_mininterval,
            smoothing=0.05,
            leave=True,
        )

    # grad accumulation
    accum = max(1, int(args.grad_accum_steps))
    can_backward_step = (
        hasattr(opt, "backward") and callable(getattr(opt, "backward"))
        and hasattr(opt, "step") and callable(getattr(opt, "step"))
    )
    if accum > 1 and not can_backward_step:
        msg = "[Warn] optimizer 不支持 opt.backward/opt.step() 分离接口，无法做 grad accumulation，将退化为每步更新。"
        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)
        accum = 1

    step = 0
    micro = 0
    ema_loss: Optional[float] = None
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

                input_ids = pack["input_ids"].astype(np.int32)          # [1,T]
                attn_mask = pack["attention_mask"].astype(np.int32)     # [1,T]
                labels = pack["labels"].astype(np.int32)                # [1,T] IGNORE outside assistant

                # remote hidden
                send_batch(sock, input_ids, attn_mask, meta={"step": step, "epoch": ep})
                hidden = recv_hidden(sock)  # [1,T,H]

                hidden_jt = jt.array(hidden)
                labels_jt = jt.array(labels)

                logits_list = medusa(hidden_jt)  # list of [B,T,V]
                total_loss = jt.float32(0.0)

                head_logs: Dict[str, Any] = {}
                B, T = labels.shape

                for k in range(args.medusa_num_heads):
                    end_idx = T - (2 + k)
                    start_label = 2 + k
                    if end_idx <= 0:
                        continue

                    logits_k = logits_list[k][:, :end_idx, :].reshape(-1, vocab_size)  # [N,V]
                    labels_k = labels_jt[:, start_label:T].reshape(-1)                  # [N]

                    loss_k, top1_k, cnt_k = masked_cross_entropy_and_top1(
                        logits_k, labels_k, IGNORE_TOKEN_ID
                    )
                    total_loss = total_loss + loss_k

                    # IMPORTANT: log python scalars only
                    head_logs[f"medusa{k}_loss"] = _scalar_float(loss_k)
                    cnt_val = _scalar_float(cnt_k)
                    head_logs[f"medusa{k}_tokens"] = cnt_val
                    if cnt_val > 0:
                        head_logs[f"medusa{k}_top1"] = _scalar_float(top1_k)

                # update
                if accum == 1:
                    opt.step(total_loss)
                else:
                    scaled = total_loss / float(accum)
                    opt.backward(scaled)
                    if micro % accum == 0:
                        opt.step()
                        if hasattr(opt, "zero_grad") and callable(getattr(opt, "zero_grad")):
                            opt.zero_grad()

                loss_val = _scalar_float(total_loss)

                # cleanup
                del logits_list, hidden_jt, labels_jt, total_loss
                jt.gc()

                dt = time.time() - t0
                tokens = args.batch_size * args.max_length
                tps = tokens / max(dt, 1e-9)

                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_val

                # stdout log
                if step % args.log_every == 0:
                    msg = f"[Train] epoch={ep} step={step} loss={loss_val:.6f} ema={ema_loss:.6f} tps={tps:.1f} lr={lr_now:.2e}"
                    if use_tqdm:
                        tqdm.write(msg)
                    else:
                        print(msg)

                # save
                if args.save_every > 0 and step % args.save_every == 0:
                    path = os.path.join(args.output_dir, f"medusa_step_{step}.pkl")
                    jt.save(medusa.state_dict(), path)
                    if use_tqdm:
                        tqdm.write(f"[Save] {path}")
                    else:
                        print(f"[Save] {path}")

                # wandb log (SAFE)
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

                # tqdm update
                if use_tqdm and pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{loss_val:.3f}",
                        ema=f"{ema_loss:.3f}",
                        tps=f"{tps:.0f}",
                        lr=f"{lr_now:.1e}",
                    )

            # epoch end save
            epoch_path = os.path.join(args.output_dir, f"medusa_epoch_{ep}.pkl")
            jt.save(medusa.state_dict(), epoch_path)
            if use_tqdm:
                tqdm.write(f"[Save] {epoch_path}")
            else:
                print(f"[Save] {epoch_path}")

            # if args.wandb and wandb is not None:
            #     try:
            #         wandb.save(epoch_path)
            #     except Exception:
            #         pass

            if args.max_steps > 0 and step >= args.max_steps:
                break

    finally:
        try:
            sock.close()
        except Exception:
            pass
        if pbar is not None:
            pbar.close()
        if args.wandb and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()