#!/usr/bin/env bash
set -euo pipefail

############################################
# One-click: Speed + Quality (single + pairwise)
# For: vicuna-7b-v1.33 baseline vs jittor-trained Medusa head
############################################

### ====== USER CONFIG (edit these) ======
GPU=2

# FastChat llm_judge dir
FASTCHAT_JUDGE_DIR="../my_judge"

# mt-bench questions
QUESTIONS="$FASTCHAT_JUDGE_DIR/data/mt_bench/question.jsonl"

# Answer output dir (FastChat expects this)
ANSWER_DIR="$FASTCHAT_JUDGE_DIR/data/mt_bench/model_answer"

# Your local vicuna path
VICUNA_LOCAL="../models/vicuna-7b-v1.33"

# Your jittor-trained medusa head
MEDUSA_PKL="medusa_jittor/checkpoints_vicuna_jtclean_compare/medusa_epoch_0.pkl"
# MEDUSA_PKL="medusa_jittor_all/checkpoints_vicuna_jt_end2end/medusa_epoch_0.pkl"
MEDUSA_HEADS=3

# Bench script path (your modified bench_mtbench.py)
BENCH_PY="my_judge/bench_mtbench.py"

# Generation settings
CONV="vicuna_v1.1"
MAX_NEW_TOKENS=256
TEMP=0.7
TOPP=0.95
WARMUP=0

# Judge model (must be allowed by your FastChat OPENAI_MODEL_LIST)
JUDGE_MODEL="gpt-4o-mini"
PARALLEL=4

# Model ids (these become filenames under model_answer/)
ID_BASELINE="vicuna133_baseline"
ID_MEDUSAJT="vicuna133_medusa_jt"

# Where to store speed json outputs
OUTROOT="../bench_runs"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$OUTROOT/$TS"
mkdir -p "$OUTDIR"

### ====== OPTIONAL: export your OpenAI proxy env here ======
# export OPENAI_API_KEY="YOUR_KEY"
# export OPENAI_API_BASE="https://your-proxy-domain/v1"

############################################
# Sanity checks
############################################
echo "== Sanity checks =="
[[ -f "$BENCH_PY" ]] || { echo "ERROR: BENCH_PY not found: $BENCH_PY"; exit 1; }
[[ -d "$FASTCHAT_JUDGE_DIR" ]] || { echo "ERROR: FASTCHAT_JUDGE_DIR not found: $FASTCHAT_JUDGE_DIR"; exit 1; }
[[ -f "$QUESTIONS" ]] || { echo "ERROR: QUESTIONS not found: $QUESTIONS"; exit 1; }
[[ -d "$VICUNA_LOCAL" ]] || { echo "ERROR: VICUNA_LOCAL not found: $VICUNA_LOCAL"; exit 1; }
[[ -f "$MEDUSA_PKL" ]] || { echo "ERROR: MEDUSA_PKL not found: $MEDUSA_PKL"; exit 1; }

# Check API env (needed for gen_judgment)
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARN: OPENAI_API_KEY is empty. Judge will fail unless you export it."
fi
if [[ -z "${OPENAI_API_BASE:-}" ]]; then
  echo "INFO: OPENAI_API_BASE is empty (ok if using official endpoint)."
fi

mkdir -p "$ANSWER_DIR"

############################################
# 1) Generate baseline answers + speed json
############################################
echo ""
echo "=== [1/6] baseline: generate answers + speed json ==="
CUDA_VISIBLE_DEVICES=$GPU python "$BENCH_PY" \
  --questions "$QUESTIONS" \
  --mode baseline \
  --model "$VICUNA_LOCAL" \
  --conv "$CONV" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" --top_p "$TOPP" \
  --warmup "$WARMUP" \
  --out_json "$OUTDIR/${ID_BASELINE}_speed.json" \
  --answer_file "$ANSWER_DIR/${ID_BASELINE}.jsonl" \
  --model_id "$ID_BASELINE"

############################################
# 2) Generate medusa answers + speed json
############################################
echo ""
echo "=== [2/6] medusa(jittor head): generate answers + speed json ==="
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
  --out_json "$OUTDIR/${ID_MEDUSAJT}_speed.json" \
  --answer_file "$ANSWER_DIR/${ID_MEDUSAJT}.jsonl" \
  --model_id "$ID_MEDUSAJT"

############################################
# 3) Judge: single
############################################
echo ""
echo "=== [3/6] judge single with ${JUDGE_MODEL} ==="
cd "$FASTCHAT_JUDGE_DIR"
yes "" | python gen_judgment.py \
  --bench-name mt_bench \
  --mode single \
  --judge-model "$JUDGE_MODEL" \
  --model-list "$ID_BASELINE" "$ID_MEDUSAJT" \
  --parallel "$PARALLEL"

echo ""
echo "=== [4/6] show single results ==="
python show_result.py \
  --bench-name mt_bench \
  --mode single \
  --judge-model "$JUDGE_MODEL" \
  --model-list "$ID_BASELINE" "$ID_MEDUSAJT" \
  | tee "$OUTDIR/${JUDGE_MODEL}_single_result.txt"

############################################
# 4) Judge: pairwise-all
############################################
echo ""
echo "=== [5/6] judge pairwise-all with ${JUDGE_MODEL} ==="
yes "" | python gen_judgment.py \
  --bench-name mt_bench \
  --mode pairwise-all \
  --judge-model "$JUDGE_MODEL" \
  --model-list "$ID_BASELINE" "$ID_MEDUSAJT" \
  --parallel "$PARALLEL"

echo ""
echo "=== [6/6] show pairwise-all results ==="
python show_result.py \
  --bench-name mt_bench \
  --mode pairwise-all \
  --judge-model "$JUDGE_MODEL" \
  --model-list "$ID_BASELINE" "$ID_MEDUSAJT" \
  | tee "$OUTDIR/${JUDGE_MODEL}_pairwise_all_result.txt"

############################################
# Summary paths
############################################
echo ""
echo "==================== DONE ===================="
echo "Speed JSON:"
echo "  $OUTDIR/${ID_BASELINE}_speed.json"
echo "  $OUTDIR/${ID_MEDUSAJT}_speed.json"
echo "Single result:"
echo "  $OUTDIR/${JUDGE_MODEL}_single_result.txt"
echo "Pairwise-all result:"
echo "  $OUTDIR/${JUDGE_MODEL}_pairwise_all_result.txt"
echo "Answers JSONL:"
echo "  $ANSWER_DIR/${ID_BASELINE}.jsonl"
echo "  $ANSWER_DIR/${ID_MEDUSAJT}.jsonl"
echo "=============================================="
