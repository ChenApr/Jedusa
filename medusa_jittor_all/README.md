# Medusa Jittor All Implementation

This directory contains the full Jittor implementation of Medusa, where both the backbone and heads are implemented in Jittor.

## Architecture

- **Backbone**: Jittor. We use a Jittor implementation of LLaMA (`modeling_llama_jt.py`) instead of PyTorch.
- **Heads**: Jittor. The Medusa heads are implemented in Jittor.

## Directory Structure

- `model/`: Jittor model definitions (`medusa_model.py`, `modeling_llama_jt.py`).
- `train/`: Training scripts.
  - `train_end2end.py`: Main script for end-to-end training.
  - `train_head_only.py`: Script for training heads only.
- `inference/`: Inference scripts (`cli.py`).

## Usage

### 1. Training

Run the end-to-end training script:

```bash
python -m medusa_jittor_all.train.train_end2end \
    --base_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path data/vicuna_2048.json \
    --output_dir checkpoints_vicuna_jt_all
```

### 2. Inference

Run the CLI demo:

```bash
python -m medusa_jittor_all.inference.cli --model checkpoints_vicuna_jt_all
```

## Note on Legacy Files

Some files copied from `medusa_jittor` are unused in this full implementation:
- `train/train.py`: Legacy hybrid training script (use `train_end2end.py` instead).
- `inference/cli_chat.py`: Legacy chat script (use `cli.py` instead).
- `scripts/verify_weights.py`: Only for PyTorch compatibility checks.
- `*.bak`: Backup files.
