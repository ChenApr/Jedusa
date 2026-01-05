# Medusa PyTorch Implementation

This directory contains the original PyTorch implementation of Medusa.

## Architecture

- **Backbone**: PyTorch (HuggingFace Transformers). We use the pre-trained Vicuna model as the frozen backbone.
- **Heads**: PyTorch. The Medusa heads (MLP layers) are implemented in PyTorch.

## Directory Structure

- `model/`: Contains the model definitions (`medusa_model.py`).
- `train/`: Contains the training script (`train.py`).
- `inference/`: Contains the inference script (`cli.py`).
- `eval/`: Contains evaluation scripts.

## Usage

### 1. Training

Run the training script from the project root:

```bash
python -m medusa.train.train \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path data/vicuna_2048.json \
    --bf16 True \
    --output_dir checkpoints_medusa \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --medusa_num_heads 3 \
    --medusa_num_layers 1
```

### 2. Inference

Run the CLI demo:

```bash
python -m medusa.inference.cli --model checkpoints_medusa
```

## Installation

Ensure you have the required dependencies installed:

```bash
pip install torch transformers
```
