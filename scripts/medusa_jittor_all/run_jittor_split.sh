#!/usr/bin/env bash
set -euo pipefail

cd ~/Jedusa

# ---- make conda activate work in non-interactive shell ----
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[FATAL] conda.sh not found. Please locate it under ~/miniconda3 or ~/anaconda3"
  exit 1
fi

conda activate jt_clean310

# ---- paths ----
MODEL="../models/vicuna-7b-v1.33"
DATA="../ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json"
OUT="medusa_jittor_all/checkpoints_vicuna_jt_split"
PORT="${PORT:-5001}"

# ---- hyperparams ----
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MEDUSA_HEADS="${MEDUSA_HEADS:-3}"
MEDUSA_LAYERS="${MEDUSA_LAYERS:-1}"
LR="${LR:-1e-3}"
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-0}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHED="${LR_SCHED:-cosine}"

# ---- wandb ----
WANDB_PROJECT="${WANDB_PROJECT:-medusa-jittor-all}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-jittor,split,no-torch}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---- clean environment base ----
BASE_ENV=(
  env -i
  HOME="$HOME"
  PATH="$CONDA_PREFIX/bin:/usr/bin:/bin"
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  WANDB_DIR="$HOME/.cache/wandb"
  WANDB_CACHE_DIR="$HOME/.cache/wandb"
  WANDB_CONFIG_DIR="$HOME/.config/wandb"
  WANDB_API_KEY="$WANDB_API_KEY"
  WANDB_PROJECT="$WANDB_PROJECT"
  WANDB_MODE="$WANDB_MODE"
  PYTHONPATH="."
)

ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARGS=(--wandb_entity "$WANDB_ENTITY")
fi

RUN_NAME="vicuna7b_medusa_jittor_split_h${MEDUSA_HEADS}_l${MEDUSA_LAYERS}_lr${LR}_len${MAX_LENGTH}_ga${GRAD_ACCUM}"

echo "[jittor-split] MODEL=$MODEL"
echo "[jittor-split] DATA=$DATA"
echo "[jittor-split] OUT=$OUT"
echo "[jittor-split] heads=$MEDUSA_HEADS layers=$MEDUSA_LAYERS lr=$LR max_length=$MAX_LENGTH grad_accum=$GRAD_ACCUM"

mkdir -p "$OUT"

# ---- 1) Jittor Worker on GPU1: Backbone Forward ----
echo "[jittor-split] Starting Jittor Worker on GPU1..."
"${BASE_ENV[@]}" CUDA_VISIBLE_DEVICES=1 \
  python -m medusa_jittor_all.train.jittor_worker \
    --base_model_name_or_path "$MODEL" \
    --port "$PORT" \
    --local_files_only \
  > "$OUT/worker.log" 2>&1 &
WORKER_PID=$!
echo "[jittor-split] Worker PID=$WORKER_PID"

cleanup() {
  echo "[jittor-split] Stopping worker..."
  kill "$WORKER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- Wait for worker ----
echo "[jittor-split] Waiting for worker on port $PORT..."
for i in {1..300}; do
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    echo "[jittor-split] Worker ready!"
    break
  fi
  sleep 1
done

# ---- 2) Jittor Trainer on GPU0: Heads Training ----
# We reuse the existing train_head_only.py from medusa_jittor but point it to the new worker
# Wait, we need to make sure train_head_only.py uses the correct MedusaHead implementation.
# The original medusa_jittor/train/train_head_only.py imports from medusa_jittor.model.medusa_head.
# We should use a version that imports from medusa_jittor_all.
# Let's create medusa_jittor_all/train/train_head_only.py if it's not correct.
# I checked earlier, it exists (copy) but imports might be wrong.
# I will assume I need to fix imports in medusa_jittor_all/train/train_head_only.py first.

echo "[jittor-split] Starting Jittor Trainer on GPU0..."
"${BASE_ENV[@]}" CUDA_VISIBLE_DEVICES=0 \
  python -m medusa_jittor_all.train.train_head_only \
    --base_model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --worker_host 127.0.0.1 \
    --worker_port "$PORT" \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --medusa_num_heads "$MEDUSA_HEADS" \
    --medusa_num_layers "$MEDUSA_LAYERS" \
    --lr "$LR" \
    --log_every "$LOG_EVERY" \
    --save_every "$SAVE_EVERY" \
    --local_files_only \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    "${ENTITY_ARGS[@]}" \
    --wandb_run_name "$RUN_NAME" \
    --wandb_tags "$WANDB_TAGS" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHED"
