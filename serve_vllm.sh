#!/usr/bin/env bash
set -euo pipefail

MODEL="../models/vicuna-7b-v1.33"
PORT=8080

vllm serve "$MODEL" \
  --served-model-name vicuna \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 2048