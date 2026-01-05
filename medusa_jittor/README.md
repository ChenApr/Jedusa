# Medusa Jittor Implementation (Hybrid Approach)

This directory contains the Jittor implementation of Medusa heads, designed to work in a hybrid manner with a PyTorch backbone.

## Architecture

- **Backbone**: PyTorch (HuggingFace Transformers). We use the pre-trained Vicuna model as the frozen backbone to generate hidden states.
- **Heads**: Jittor. The Medusa heads (MLP layers) are implemented and trained in Jittor.

## Directory Structure

- `model/`: Contains the Jittor model definitions (`medusa_head.py`).
- `train/`: Contains the hybrid training script (`train.py`).
- `scripts/`: Helper scripts (e.g., `verify_weights.py`).

## Usage

### 1. Training

Run the training script from the project root:

```bash
python -m medusa_jittor.train.train \
    --base_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path data/vicuna_2048.json \
    --output_dir checkpoints_jittor
```

This will:
1. Load the Vicuna model in PyTorch (frozen).
2. Initialize Medusa heads in Jittor.
3. Train the heads using the hidden states from Vicuna.
4. Save checkpoints in `.pkl` (Jittor) and `.npz` (PyTorch-compatible) formats.

### 2. Verifying Weights

You can verify that the saved `.npz` weights are compatible with the original PyTorch Medusa implementation:

```bash
python medusa_jittor/scripts/verify_weights.py checkpoints_jittor/medusa_head_epoch_0_pt.npz
```

### 3. Loading into PyTorch for Inference

To use the trained heads in the original Medusa inference pipeline, you can load the `.npz` file:

```python
import torch
import numpy as np
from medusa.model.medusa_model import MedusaModel

# Load base model
model = MedusaModel.from_pretrained("lmsys/vicuna-7b-v1.3")

# Load weights from NPZ
weights = np.load("checkpoints_jittor/medusa_head_epoch_0_pt.npz")
state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}

# Load into model
model.medusa_head.load_state_dict(state_dict, strict=False)
```

### 4. Inference (CLI Chat)

You can chat with the model using the CLI interface:

```bash
python -m medusa_jittor.inference.cli_chat \
    --base_model ../models/vicuna-7b-v1.3 \
    --medusa_weights checkpoints_vicuna_jtclean_compare/medusa_epoch_0.pkl \
    --medusa_num_heads 3
```

Arguments:
- `--base_model`: Path to the Vicuna base model.
- `--medusa_weights`: Path to the trained Medusa heads (supports `.pkl`, `.npz`, `.pt`, `.bin`).
- `--load-in-8bit` / `--load-in-4bit`: Optional quantization for the base model.

> **Tip**: Use `--style rich` for a better looking chat interface (requires `rich` installed).


