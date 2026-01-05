import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import jittor as jt
jt.flags.use_cuda = 1
# jt.flags.fp16 = 1 # Try enabling this

import torch
import numpy as np
import json
import gc
from medusa_jittor_all.model.medusa_model import MedusaModel

base_model_path = "../models/vicuna-7b-v1.33"

def print_mem(tag=""):
    jt.gc()
    gc.collect()
    jt.display_memory_info()
    print(f"[{tag}] Memory checked.")

print_mem("Start")

print("Initializing MedusaModel...")
model = MedusaModel.from_pretrained(
    base_model_path,
    medusa_num_heads=3,
    torch_dtype=jt.float16
)
print_mem("After Init")

# Load weights manually to monitor memory
path = base_model_path
shard_files = []
index_file = os.path.join(path, "pytorch_model.bin.index.json")
if os.path.exists(index_file):
    with open(index_file, "r") as f:
        index = json.load(f)
    shard_files = sorted(list(set(index["weight_map"].values())))
    shard_files = [os.path.join(path, f) for f in shard_files]
else:
    shard_files = [os.path.join(path, "pytorch_model.bin")]

for i, shard_file in enumerate(shard_files):
    print(f"Loading shard {i}: {shard_file}")
    state_dict = torch.load(shard_file, map_location="cpu")
    
    jt_state_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32 or v.dtype == torch.float16:
            jt_state_dict[k] = jt.array(v.to(torch.float16).numpy())
        else:
            jt_state_dict[k] = jt.array(v.numpy())
    
    model.load_state_dict(jt_state_dict, strict=False)
    del state_dict
    del jt_state_dict
    print_mem(f"After Shard {i}")

print("Model Loaded.")
print_mem("Final")
