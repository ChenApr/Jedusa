#!/usr/bin/env bash
set -euo pipefail

FASTCHAT_JUDGE_DIR="../my_judge"
QUESTIONS="$FASTCHAT_JUDGE_DIR/data/mt_bench/question.jsonl"
ANSWER_DIR="$FASTCHAT_JUDGE_DIR/data/mt_bench/model_answer"
CONV="vicuna_v1.1"
GPU=0

MAX_NEW_TOKENS=256     # 先用 256 跑通
TEMP=0.7
TOPP=0.95

VICUNA_V13="../models/vicuna-7b-v1.3"
MEDUSA1="../models/medusa-vicuna-7b-v1.3"
MEDUSA2="../models/medusa-v1.0-vicuna-7b-v1.5"   # 可选

ID_BASELINE="baseline_v13"
ID_MEDUSA1="medusa1_v13"
ID_MEDUSA2="medusa2_v15"

mkdir -p "$ANSWER_DIR"
rm -f "$ANSWER_DIR/${ID_BASELINE}.jsonl" "$ANSWER_DIR/${ID_MEDUSA1}.jsonl" "$ANSWER_DIR/${ID_MEDUSA2}.jsonl"

echo "=== baseline answers ==="
CUDA_VISIBLE_DEVICES=$GPU python my_judge/bench_mtbench.py \
  --questions "$QUESTIONS" \
  --mode baseline \
  --model "$VICUNA_V13" \
  --conv "$CONV" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" --top_p "$TOPP" \
  --warmup 0 \
  --answer_file "$ANSWER_DIR/${ID_BASELINE}.jsonl" \
  --model_id "$ID_BASELINE"

echo "=== medusa1 answers ==="
CUDA_VISIBLE_DEVICES=$GPU python my_judge/bench_mtbench.py \
  --questions "$QUESTIONS" \
  --mode medusa \
  --model "$MEDUSA1" \
  --base-model "$VICUNA_V13" \
  --conv "$CONV" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" --top_p "$TOPP" \
  --warmup 0 \
  --answer_file "$ANSWER_DIR/${ID_MEDUSA1}.jsonl" \
  --model_id "$ID_MEDUSA1"

# 如果你要也评 medusa2（探索性）
echo "=== medusa2 answers (optional) ==="
CUDA_VISIBLE_DEVICES=$GPU python bench_mtbench.py \
  --questions "$QUESTIONS" \
  --mode medusa \
  --model "$MEDUSA2" \
  --conv "$CONV" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP" --top_p "$TOPP" \
  --warmup 0 \
  --answer_file "$ANSWER_DIR/${ID_MEDUSA2}.jsonl" \
  --model_id "$ID_MEDUSA2"

cd "$FASTCHAT_JUDGE_DIR"

yes "" | python gen_judgment.py \
  --bench-name mt_bench \
  --mode single \
  --judge-model gpt-4 \
  --model-list "$ID_BASELINE" "$ID_MEDUSA1" "$ID_MEDUSA2" \
  --parallel 4

python show_result.py \
  --bench-name mt_bench \
  --mode single \
  --judge-model gpt-4 \
  --model-list "$ID_BASELINE" "$ID_MEDUSA1" "$ID_MEDUSA2"
