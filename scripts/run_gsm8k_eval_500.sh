#!/usr/bin/env bash
set -euo pipefail

############################################
# One-click: GSM8K Evaluation (Baseline vs Medusa)
############################################

### ====== USER CONFIG (edit these) ======
GPU=1

# Your local vicuna path
VICUNA_LOCAL="../models/vicuna-7b-v1.33"

# Your jittor-trained medusa head
MEDUSA_PKL="medusa_jittor/checkpoints_vicuna_jtclean_compare/medusa_epoch_0.pkl"
MEDUSA_HEADS=3

# Scripts
BENCH_PY="../my_judge/bench_gsm8k.py"
DOWNLOAD_PY="../my_judge/download_gsm8k.py"
EVAL_PY="my_judge/eval_gsm8k.py"

# Data
LOCAL_GSM8K="../grade-school-math/grade_school_math/data/test.jsonl"
DATA_DIR="data/gsm8k"
QUESTIONS="$DATA_DIR/test_bench.jsonl"
GROUND_TRUTH="$DATA_DIR/test_ground_truth.jsonl"

# Generation settings
CONV="vicuna_v1.1"
MAX_NEW_TOKENS=512
TEMP=0.0  # Greedy for math
TOPP=1.0
WARMUP=0
LIMIT=500
MAX_CTX=2560

# Output
OUTROOT="../bench_runs_gsm8k"
# TS="$(date +%Y%m%d_%H%M%S)"
# Use the latest existing directory if available to resume/append, or create new
# But user wants to skip baseline, so we should probably reuse the previous dir if we want to compare?
# Actually, if we skip baseline generation, we might still want to evaluate it if the file exists.
# Let's just keep creating a new dir for simplicity, but comment out baseline generation.
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$OUTROOT/$TS"
mkdir -p "$OUTDIR"

ID_BASELINE="vicuna133_baseline"
ID_MEDUSAJT="vicuna133_medusa_jt"

############################################
# Sanity checks & Data Prep
############################################
echo "== Sanity checks =="
[[ -f "$BENCH_PY" ]] || { echo "ERROR: BENCH_PY not found: $BENCH_PY"; exit 1; }
[[ -d "$VICUNA_LOCAL" ]] || { echo "ERROR: VICUNA_LOCAL not found: $VICUNA_LOCAL"; exit 1; }
[[ -f "$MEDUSA_PKL" ]] || { echo "ERROR: MEDUSA_PKL not found: $MEDUSA_PKL"; exit 1; }

if [[ ! -f "$QUESTIONS" ]] || [[ ! -f "$GROUND_TRUTH" ]]; then
    echo "GSM8K data not found. Preparing..."
    python "$DOWNLOAD_PY" --local-path "$LOCAL_GSM8K"
fi

[[ -f "$QUESTIONS" ]] || { echo "ERROR: QUESTIONS not found after download: $QUESTIONS"; exit 1; }

############################################
# 1) Generate baseline answers (SKIPPED)
############################################
echo ""
echo "=== [1/4] baseline: generate answers (SKIPPED) ==="
CUDA_VISIBLE_DEVICES=$GPU python "$BENCH_PY" \
  --questions "$QUESTIONS" \
  --mode baseline \
  --model "$VICUNA_LOCAL" \
  --conv "$CONV" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" --top_p "$TOPP" \
  --warmup "$WARMUP" \
  --limit "$LIMIT" \
  --max_ctx "$MAX_CTX" \
  --out_json "$OUTDIR/${ID_BASELINE}_speed.json" \
  --answer_file "$OUTDIR/${ID_BASELINE}.jsonl" \
  --model_id "$ID_BASELINE"

############################################
# 2) Generate medusa answers
############################################
echo ""
echo "=== [2/4] medusa(jittor head): generate answers ==="
CUDA_VISIBLE_DEVICES=$GPU python "$BENCH_PY" \
  --questions "$QUESTIONS" \
  --mode medusa \
  --model "$VICUNA_LOCAL" \
  --medusa_weights "$MEDUSA_PKL" \
  --medusa_num_heads "$MEDUSA_HEADS" \
  --conv "$CONV" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" --top_p "$TOPP" \
  --warmup "$WARMUP" \
  --limit "$LIMIT" \
  --max_ctx "$MAX_CTX" \
  --out_json "$OUTDIR/${ID_MEDUSAJT}_speed.json" \
  --answer_file "$OUTDIR/${ID_MEDUSAJT}.jsonl" \
  --model_id "$ID_MEDUSAJT"

############################################
# 3) Evaluate Baseline (SKIPPED)
############################################
echo ""
echo "=== [3/4] Evaluate Baseline (SKIPPED) ==="
python "$EVAL_PY" \
  --answer-file "$OUTDIR/${ID_BASELINE}.jsonl" \
  --ground-truth-file "$GROUND_TRUTH"

############################################
# 4) Evaluate Medusa
############################################
echo ""
echo "=== [4/4] Evaluate Medusa ==="
python "$EVAL_PY" \
  --answer-file "$OUTDIR/${ID_MEDUSAJT}.jsonl" \
  --ground-truth-file "$GROUND_TRUTH"

############################################
# Summary
############################################
echo ""
echo "==================== DONE ===================="
echo "Results saved in $OUTDIR"
