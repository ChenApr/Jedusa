import torch
import torch.nn as nn
import numpy as np
import argparse
import os

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, medusa_num_heads, medusa_num_layers):
        super().__init__()
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size)] * medusa_num_layers),
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )

    def forward(self, x):
        results = []
        for head in self.medusa_head:
            results.append(head(x))
        return results

def verify(args):
    print(f"Loading weights from {args.npz_path}...")
    data = np.load(args.npz_path)
    
    # Infer config from weights if possible, or use defaults/args
    # keys example: medusa_head.0.0.linear.weight
    
    # Check keys
    keys = list(data.keys())
    print(f"Found {len(keys)} keys in NPZ.")
    
    # Simple heuristic to guess config
    # medusa_head.{head_idx}.{layer_idx/linear}.weight
    
    max_head_idx = 0
    for k in keys:
        parts = k.split('.')
        if parts[0] == "medusa_head" and parts[1].isdigit():
            max_head_idx = max(max_head_idx, int(parts[1]))
            
    medusa_num_heads = max_head_idx + 1
    print(f"Detected medusa_num_heads: {medusa_num_heads}")
    
    # We need hidden_size and vocab_size to initialize model
    # medusa_head.0.1.weight (Linear layer at end) -> [vocab_size, hidden_size]
    # medusa_head.0.0.linear.weight (ResBlock) -> [hidden_size, hidden_size]
    
    # Find the final linear layer of head 0
    # It should be the last element in the sequential.
    # If num_layers=1, sequential is [ResBlock, Linear]. ResBlock is 0, Linear is 1.
    # So medusa_head.0.1.weight
    
    # Let's find a key that looks like the final linear
    vocab_size = 0
    hidden_size = 0
    
    for k in keys:
        if "medusa_head.0." in k and ".weight" in k and "linear" not in k:
            # This is likely the final linear layer: medusa_head.0.X.weight
            w = data[k]
            vocab_size = w.shape[0]
            hidden_size = w.shape[1]
            print(f"Inferred vocab_size: {vocab_size}, hidden_size: {hidden_size}")
            break
            
    if vocab_size == 0:
        print("Could not infer sizes. Using defaults/args.")
        vocab_size = 32000
        hidden_size = 4096

    model = MedusaHead(hidden_size, vocab_size, medusa_num_heads, args.medusa_num_layers)
    
    # Load weights
    print("Loading state dict...")
    state_dict = {}
    for k in keys:
        state_dict[k] = torch.from_numpy(data[k])
        
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("SUCCESS: All weights loaded perfectly!")
    else:
        print(f"WARNING: Loading finished with issues.")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str, help="Path to the .npz file")
    parser.add_argument("--medusa_num_layers", type=int, default=1)
    args = parser.parse_args()
    verify(args)
