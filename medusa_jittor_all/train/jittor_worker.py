import os
import json
import socket
import struct
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import jittor as jt

# Use our Jittor model implementation
from medusa_jittor_all.model.medusa_model import MedusaModel

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data

def recv_batch(sock):
    # Read frame length
    try:
        len_bytes = sock.recv(4)
        if not len_bytes:
            return None
        if len(len_bytes) < 4:
            len_bytes += recv_exact(sock, 4 - len(len_bytes))
    except ConnectionError:
        return None

    frame_len = struct.unpack("!I", len_bytes)[0]
    
    # Read header length
    header_len_bytes = recv_exact(sock, 4)
    header_len = struct.unpack("!I", header_len_bytes)[0]
    
    # Read header
    header_bytes = recv_exact(sock, header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    
    # Read blobs
    blobs = {}
    for name in ["input_ids", "attention_mask"]:
        info = header[name]
        nbytes = info["nbytes"]
        data = recv_exact(sock, nbytes)
        blobs[name] = np.frombuffer(data, dtype=info["dtype"]).reshape(info["shape"])
        
    return blobs, header["meta"]

def send_hidden(sock, hidden_np):
    # hidden_np: [bs, seq_len, hidden_size]
    header = {
        "hidden": {
            "dtype": str(hidden_np.dtype),
            "shape": list(hidden_np.shape),
            "nbytes": hidden_np.nbytes,
            "offset": 0
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack("!I", len(header_bytes)) + header_bytes + hidden_np.tobytes()
    sock.sendall(struct.pack("!I", len(payload)) + payload)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    print(f"[Worker] Initializing Jittor on GPU...")
    jt.flags.use_cuda = 1

    print(f"[Worker] Loading model from {args.base_model_name_or_path}...")
    # Load model in fp16
    model = MedusaModel.from_pretrained(
        args.base_model_name_or_path,
        medusa_num_heads=1, # Dummy, we only need backbone
        medusa_num_layers=1,
        local_files_only=args.local_files_only
    )
    
    # Convert to fp16
    print("[Worker] Converting to float16...")
    def to_half(module):
        for k, v in module.__dict__.items():
            if isinstance(v, jt.Var):
                if v.dtype == jt.float32:
                    module.__dict__[k] = v.float16()
        for idx, m in enumerate(module.modules()):
            if m is not module:
                to_half(m)
    to_half(model)
    
    # Force gc
    jt.gc()
    
    # Setup socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"[Worker] Listening on port {args.port}...")

    while True:
        conn, addr = server.accept()
        print(f"[Worker] Connected by {addr}")
        try:
            while True:
                res = recv_batch(conn)
                if res is None:
                    break
                blobs, meta = res
                
                input_ids = jt.array(blobs["input_ids"])
                attention_mask = jt.array(blobs["attention_mask"])
                
                # Forward
                with jt.no_grad():
                    # We need to access the base model directly to get hidden states
                    # MedusaModel -> base_model -> model (LlamaModel)
                    # LlamaModel returns (hidden_states, ...)
                    
                    # MedusaModel.forward calls self.base_model.model(...)
                    # Let's call it directly
                    outputs = model.base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    hidden_states = outputs[0]
                    
                    # Sync and convert to numpy
                    hidden_np = hidden_states.numpy()
                    
                send_hidden(conn, hidden_np)
                
                # Cleanup
                del input_ids, attention_mask, outputs, hidden_states
                # jt.gc() # Optional, might slow down
                
        except Exception as e:
            print(f"[Worker] Error: {e}")
        finally:
            conn.close()
            print("[Worker] Connection closed")

if __name__ == "__main__":
    main()
