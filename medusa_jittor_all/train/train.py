import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import jittor as jt
from jittor import nn

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from medusa_jittor.model.medusa_head import MedusaHead


# --------------------------
# Dataset (PyTorch DataLoader, no jittor.dataset dependency)
# --------------------------
class ShareGPTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # ShareGPT 常见是 list
            self.data = obj if isinstance(obj, list) else obj.get("data", [])
        except Exception as e:
            print(f"[Dataset] Failed to load {data_path}: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _format_item(item: Dict[str, Any]) -> str:
        # 兼容 conversations / text / 其他
        if isinstance(item, dict) and "conversations" in item:
            parts = []
            for msg in item["conversations"]:
                role = msg.get("from", "")
                content = msg.get("value", "")
                if role == "human":
                    parts.append(f"USER: {content}")
                elif role == "gpt":
                    parts.append(f"ASSISTANT: {content} </s>")
                else:
                    parts.append(str(content))
            return " ".join(parts)
        if isinstance(item, dict) and "text" in item:
            return str(item["text"])
        return str(item)

    def __getitem__(self, idx: int):
        text = self._format_item(self.data[idx])

        tok = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok["input_ids"][0]          # [T]
        attention_mask = tok["attention_mask"][0]  # [T]
        labels = input_ids.clone()               # [T] 这里按你的原逻辑：labels=input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }


# --------------------------
# Train
# --------------------------
def train(args):
    # ---- Jittor device ----
    if args.device == "cuda":
        jt.flags.use_cuda = 1
        print("[Jittor] Using CUDA")
    else:
        jt.flags.use_cuda = 0
        print("[Jittor] Using CPU")

    # ---- Torch device ----
    torch_device = "cuda:1" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch_dtype = torch.float16 if (torch_device != "cpu") else torch.float32
    print(f"[Torch] device={torch_device}, dtype={torch_dtype}")

    if torch_device.startswith("cuda"):
        torch.cuda.set_device(int(torch_device.split(":")[1]))

    # ---- Tokenizer ----
    print(f"[Tokenizer] Loading from {args.base_model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=False, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Backbone (PyTorch, frozen) ----
    print(f"[Backbone] Loading from {args.base_model_name_or_path} ...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,                 # 关键：别 auto
            local_files_only=args.local_files_only,
            use_safetensors=args.use_safetensors,
            low_cpu_mem_usage=True,
        )
        base_model.to(torch_device)          # 关键：整模型上 GPU1
    except ValueError as e:
        # 典型：torch<2.6 时 transformers 拦 torch.load
        print("\n[Backbone] Load failed.")
        print("If you see the CVE-2025-32434 / torch.load safety error, upgrade torch>=2.6 or use safetensors.")
        raise

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    # inner backbone
    backbone = base_model.model if hasattr(base_model, "model") else base_model.base_model

    hidden_size = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size
    print(f"[Backbone] hidden_size={hidden_size}, vocab_size={vocab_size}, tokenizer_size={len(tokenizer)}")

    # ---- Medusa head (Jittor, trainable) ----
    print("[Medusa] Init head ...")
    medusa_head = MedusaHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
    )
    optimizer = nn.AdamW(medusa_head.parameters(), lr=args.lr)

    # ---- DataLoader ----
    dataset = ShareGPTDataset(args.data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    print(f"[Data] num_samples={len(dataset)}")

    # ---- Train loop ----
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in loader:
            global_step += 1
            if args.max_steps > 0 and global_step > args.max_steps:
                break

            input_ids_pt = batch["input_ids"].to(torch_device)
            attn_mask_pt = batch["attention_mask"].to(torch_device)
            labels_pt = batch["labels"]  # keep on CPU for slicing -> jt

            # clamp if tokenizer bigger than vocab (安全兜底)
            if (input_ids_pt.max().item() >= vocab_size):
                input_ids_pt = torch.clamp(input_ids_pt, 0, vocab_size - 1)
                labels_pt = input_ids_pt.detach().cpu()

            with torch.no_grad():
                out = backbone(input_ids=input_ids_pt, attention_mask=attn_mask_pt)
                hidden = out.last_hidden_state  # [B,T,H]

            # move hidden to CPU numpy -> Jittor
            # Ensure float32 for Jittor compatibility (weights are float32)
            hidden_np = hidden.detach().float().cpu().numpy()
            labels_np = labels_pt.numpy().astype(np.int32)

            # free torch graph ASAP
            del out, hidden
            if torch_device == "cuda":
                torch.cuda.empty_cache()

            hidden_jt = jt.array(hidden_np)
            labels_jt = jt.array(labels_np)

            # forward medusa heads
            logits_list = medusa_head(hidden_jt)  # list of [B,T,V]

            # medusa loss: head k predicts token offset (2+k)
            # 参照你原来的对齐方式：logits[:, :- (2+k)] vs labels[:, (2+k):]
            loss = 0.0
            B, T = labels_np.shape

            for k in range(args.medusa_num_heads):
                end_idx = T - (2 + k)
                start_label = 2 + k
                if end_idx <= 0:
                    continue

                logits_k = logits_list[k][:, 0:end_idx, :]          # [B, end, V]
                labels_k = labels_jt[:, start_label:T]               # [B, end]

                logits_k = logits_k.reshape(-1, logits_k.shape[-1])  # [B*end, V]
                labels_k = labels_k.reshape(-1)                      # [B*end]

                loss += nn.cross_entropy_loss(logits_k, labels_k)

            optimizer.step(loss)

            loss_val = float(loss.item())
            # cleanup
            del logits_list, hidden_jt, labels_jt, loss
            jt.gc()

            if global_step % args.log_every == 0:
                print(f"[Train] epoch={epoch} step={global_step} loss={loss_val:.6f}")

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"medusa_head_step_{global_step}.pkl")
                jt.save(medusa_head.state_dict(), save_path)
                print(f"[Save] {save_path}")

        # epoch end save
        save_path = os.path.join(args.output_dir, f"medusa_head_epoch_{epoch}.pkl")
        jt.save(medusa_head.state_dict(), save_path)
        print(f"[Save] {save_path}")

        if args.max_steps > 0 and global_step > args.max_steps:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_name_or_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--medusa_num_heads", type=int, default=5)
    p.add_argument("--medusa_num_layers", type=int, default=1)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=0)   # 0=only epoch end
    p.add_argument("--max_steps", type=int, default=0)    # 0=no limit

    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--use_safetensors", action="store_true")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()