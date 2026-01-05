#!/usr/bin/env bash
set -euo pipefail

INPUT="../ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json"
OUT="vicuna_2048.json"

python create_data_fast.py \
  --input-filename "$INPUT" \
  --output-filename "$OUT" \
  --base-url "http://localhost:8080" \
  --model "vicuna" \
  --concurrency 128 \
  --max-new-tokens 256 \
  --pick first