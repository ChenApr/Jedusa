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
OUT="medusa_jittor/checkpoints_vicuna_jtclean_small"
PORT="${PORT:-5000}"

# ---- compare-aligned hyperparams (match your torch reference as much as possible) ----
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"              # head-only version: keep 1 for stability
MAX_LENGTH="${MAX_LENGTH:-2048}"           # match torch: 2048
MEDUSA_HEADS="${MEDUSA_HEADS:-3}"          # match torch: 3
MEDUSA_LAYERS="${MEDUSA_LAYERS:-1}"        # match torch: 1
LR="${LR:-1e-3}"                           # match torch: 1e-3
LOG_EVERY="${LOG_EVERY:-100}"              # match torch logging_steps=100
SAVE_EVERY="${SAVE_EVERY:-0}"              # match save_strategy="epoch" -> 0 here (only epoch end)

# Optional: if your train_head_only.py supports these, enable them for better comparability
GRAD_ACCUM="${GRAD_ACCUM:-16}"             # match torch gradient_accumulation_steps=16 (requires code support)
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"        # match torch warmup_ratio (requires code support)
LR_SCHED="${LR_SCHED:-cosine}"             # match torch lr_scheduler_type (requires code support)

# ---- wandb ----
WANDB_PROJECT="${WANDB_PROJECT:-medusa-jittor-compare}"
WANDB_MODE="${WANDB_MODE:-online}"         # online/offline/disabled
WANDB_DIR="${WANDB_DIR:-$HOME/.cache/wandb}"
WANDB_TAGS="${WANDB_TAGS:-jittor,split,compare}"

# If you already did `wandb login`, you usually DON'T need WANDB_API_KEY in env.
# Keep it empty to avoid leaking or overriding.
WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---- clean environment base (keep only what we need) ----
BASE_ENV=(
  env -i
  HOME="$HOME"
  PATH="$CONDA_PREFIX/bin:/usr/bin:/bin"
  PYTHONNOUSERSITE=1
  WANDB_DIR="$WANDB_DIR"
  WANDB_CACHE_DIR="$WANDB_DIR"
  WANDB_CONFIG_DIR="$HOME/.config/wandb"
  WANDB_API_KEY="$WANDB_API_KEY"
  WANDB_PROJECT="$WANDB_PROJECT"
  WANDB_MODE="$WANDB_MODE"
)

ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARGS=(--wandb_entity "$WANDB_ENTITY")
fi

RUN_NAME="vicuna7b_medusa_head_jittor_cmp_h${MEDUSA_HEADS}_l${MEDUSA_LAYERS}_lr${LR}_len${MAX_LENGTH}_ga${GRAD_ACCUM}"

echo "[compare] MODEL=$MODEL"
echo "[compare] DATA=$DATA"
echo "[compare] OUT=$OUT"
echo "[compare] heads=$MEDUSA_HEADS layers=$MEDUSA_LAYERS lr=$LR max_length=$MAX_LENGTH grad_accum=$GRAD_ACCUM"
echo "[compare] wandb project=$WANDB_PROJECT mode=$WANDB_MODE run=$RUN_NAME tags=$WANDB_TAGS"

mkdir -p "$OUT"

# ---- 1) torch worker on GPU1: backbone forward only ----
"${BASE_ENV[@]}" CUDA_VISIBLE_DEVICES=1 \
  python -m medusa_jittor.train.torch_worker \
    --base_model_name_or_path "$MODEL" \
    --port "$PORT" \
    --local_files_only \
  &
WORKER_PID=$!
echo "[compare] torch worker pid=$WORKER_PID"

cleanup() {
  echo "[compare] stopping worker..."
  kill "$WORKER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- wait worker port ready ----
echo "[compare] waiting worker on 127.0.0.1:$PORT ..."
for i in {1..120}; do
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    echo "[compare] worker ready"
    break
  fi
  sleep 0.5
done

# ---- 2) jittor trainer on GPU0 ----
rm -rf ~/.cache/jittor/jt1.3.10

"${BASE_ENV[@]}" CUDA_VISIBLE_DEVICES=0 \
  python -m medusa_jittor.train.train_head_only \
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
    --lr_scheduler_type "$LR_SCHED" \
    --max_step 10