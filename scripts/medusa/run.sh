#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# ---- make conda activate work in non-interactive shell ----
# Adjust the path to your conda installation
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Activate your PyTorch environment
# conda activate medusa 

# ---- paths ----
# Adjust these paths to your local setup
MODEL="lmsys/vicuna-7b-v1.3"
DATA="data/vicuna_2048.json"
OUT="checkpoints_medusa"

# ---- hyperparams ----
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-3}"
MEDUSA_HEADS="${MEDUSA_HEADS:-3}"
MEDUSA_LAYERS="${MEDUSA_LAYERS:-1}"

echo "Starting Original Medusa Training..."
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Output: $OUT"

python -m medusa.train.train \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    --bf16 True \
    --output_dir "$OUT" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --medusa_num_heads "$MEDUSA_HEADS" \
    --medusa_num_layers "$MEDUSA_LAYERS"
