# medusa_jittor/train/torch_worker.py
import os
import json
import argparse
import socketserver
import struct
import numpy as np

# 关键：torch 只在这个进程里 import
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_msg(sock):
    # 协议：4字节长度 + payload(JSON header + raw arrays)
    n = struct.unpack("!I", recv_exact(sock, 4))[0]
    payload = recv_exact(sock, n)
    header_len = struct.unpack("!I", payload[:4])[0]
    header = json.loads(payload[4:4+header_len].decode("utf-8"))
    blob = memoryview(payload[4+header_len:])

    # 还原 input_ids / attention_mask
    def take_array(name):
        info = header[name]
        offset, nbytes = info["offset"], info["nbytes"]
        arr = np.frombuffer(blob[offset:offset+nbytes], dtype=info["dtype"]).reshape(info["shape"])
        return arr

    return header, take_array("input_ids"), take_array("attention_mask")


def send_msg(sock, header: dict, arrays: dict):
    # arrays: name -> np.ndarray
    meta = {}
    blobs = []
    offset = 0
    for k, v in arrays.items():
        v = np.ascontiguousarray(v)
        b = v.tobytes()
        meta[k] = {
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "offset": offset,
            "nbytes": len(b),
        }
        blobs.append(b)
        offset += len(b)

    header_out = {"meta": header.get("meta", {}), **meta}
    header_bytes = json.dumps(header_out).encode("utf-8")
    payload = struct.pack("!I", len(header_bytes)) + header_bytes + b"".join(blobs)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                header, input_ids_np, attn_np = recv_msg(self.request)
            except Exception:
                break

            # 前向
            # input_ids = torch.from_numpy(input_ids_np).to(self.server.torch_device)
            input_ids = torch.from_numpy(input_ids_np.copy()).to(self.server.torch_device)
            attn_mask = torch.from_numpy(attn_np.copy()).to(self.server.torch_device)


            with torch.no_grad():
                out = self.server.backbone(input_ids=input_ids, attention_mask=attn_mask)
                hidden = out.last_hidden_state  # [B,T,H]

            # 回传：为了兼容 jittor，建议 float32
            hidden_np = hidden.detach().float().cpu().numpy()
            send_msg(self.request, header, {"hidden": hidden_np})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--base_model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--use_safetensors", action="store_true")
    args = p.parse_args()

    # 固定 torch 只用当前可见的 0 号卡（你会用 CUDA_VISIBLE_DEVICES=1 启动它）
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
    print(f"[TorchWorker] device={torch_device}, dtype={torch_dtype}")

    tok = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, use_fast=False, local_files_only=args.local_files_only
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=None,
        local_files_only=args.local_files_only,
        use_safetensors=args.use_safetensors,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    backbone = model.model if hasattr(model, "model") else model.base_model
    backbone.to(torch_device)

    with socketserver.TCPServer((args.host, args.port), Handler) as srv:
        srv.torch_device = torch_device
        srv.backbone = backbone
        print(f"[TorchWorker] listening on {args.host}:{args.port}")
        srv.serve_forever()


if __name__ == "__main__":
    main()