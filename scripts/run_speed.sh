# CUDA_VISIBLE_DEVICES=0 python run_fig3_right.py \
#   --questions ../FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
#   --out-dir ../fig3_right_results \
#   --repeats 10 \
#   --conv vicuna_v1.1 \
#   --max_new_tokens 256 \
#   --temperature 0.7 \
#   --warmup 3 \
#   --progress \
#   --baseline-model ../models/vicuna-7b-v1.3 \
#   --medusa2-model ../models/medusa-v1.0-vicuna-7b-v1.5
#!/usr/bin/env bash
set -euo pipefail

QUESTIONS="../my_judge/data/mt_bench/question.jsonl"
CONV="vicuna_v1.1"
MAX_NEW_TOKENS=256
TEMP=0.7
TOPP=0.95
GPU=0
RUNS=10

# v1.3 baseline + Medusa-1 head(v1.3)
VICUNA_V13="../models/vicuna-7b-v1.3"
MEDUSA1="../models/medusa-vicuna-7b-v1.3"

# Medusa-2 full model (v1.5) —— 注意：这里仍然会和 v1.3 baseline 比（探索性）
MEDUSA2="../models/medusa-v1.0-vicuna-7b-v1.5"

TS="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-./bench_runs}"
OUTDIR="$OUTROOT/$TS"
LOGDIR="$OUTDIR/logs"
JSONDIR="$OUTDIR/json"
mkdir -p "$LOGDIR" "$JSONDIR"

CSV="$OUTDIR/results.csv"
echo "ts,exp,run,mode,model,base_model,max_new_tokens,temp,top_p,tokens_per_second,total_tokens,total_time_s,json_path,rc" > "$CSV"

export PYTORCH_ALLOC_CONF=expandable_segments:True

run_one () {
  local exp="$1"   # baseline_v13 | medusa1 | medusa2
  local run="$2"

  local mode model base_model extra_args
  mode=""
  model=""
  base_model=""
  extra_args=""

  if [[ "$exp" == "baseline_v13" ]]; then
    mode="baseline"
    model="$VICUNA_V13"
  elif [[ "$exp" == "medusa1" ]]; then
    mode="medusa"
    model="$MEDUSA1"
    base_model="$VICUNA_V13"
    extra_args="--base-model $base_model"
  elif [[ "$exp" == "medusa2" ]]; then
    mode="medusa"
    model="$MEDUSA2"
  else
    echo "unknown exp: $exp" >&2
    exit 1
  fi

  local logfile="$LOGDIR/${exp}_run$(printf "%02d" "$run").log"
  local jsonfile="$JSONDIR/${exp}_run$(printf "%02d" "$run").json"

  echo "=== [$exp] run $run @ $(date -Is) ===" > "$logfile"
  nvidia-smi >> "$logfile" || true
  echo "" >> "$logfile"

  set +e
  CUDA_VISIBLE_DEVICES=$GPU python my_judge/bench_mtbench.py \
    --questions "$QUESTIONS" \
    --mode "$mode" \
    --model "$model" \
    $extra_args \
    --conv "$CONV" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMP" \
    --top_p "$TOPP" \
    --out_json "$jsonfile" 2>&1 | tee -a "$logfile"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "$(date -Is),$exp,$run,$mode,$model,$base_model,$MAX_NEW_TOKENS,$TEMP,$TOPP,NA,NA,NA,$jsonfile,$rc" >> "$CSV"
    return 0
  fi

  local tps tokens tsec
  tps=$(python - <<PY
import json
d=json.load(open("$jsonfile"))
print(d["tokens_per_second"])
PY
)
  tokens=$(python - <<PY
import json
d=json.load(open("$jsonfile"))
print(d["total_tokens"])
PY
)
  tsec=$(python - <<PY
import json
d=json.load(open("$jsonfile"))
print(d["total_time_s"])
PY
)

  echo "$(date -Is),$exp,$run,$mode,$model,$base_model,$MAX_NEW_TOKENS,$TEMP,$TOPP,$tps,$tokens,$tsec,$jsonfile,$rc" >> "$CSV"
}

for exp in baseline_v13 medusa1 medusa2; do
  for run in $(seq 1 $RUNS); do
    run_one "$exp" "$run"
  done
done

# Summarize overall + per-category speedup
CSV_PATH="$CSV" python - <<'PY'
import csv, json, os, statistics
from collections import defaultdict

rows = list(csv.DictReader(open(os.environ["CSV_PATH"], newline="")))

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

exp_runs = defaultdict(list)
for r in rows:
    if r["rc"] != "0":
        continue
    exp_runs[r["exp"]].append(load_json(r["json_path"]))

def mean(xs): 
    return statistics.mean(xs) if xs else float("nan")

def summarize(exp):
    rs = exp_runs.get(exp, [])
    overall = [x["tokens_per_second"] for x in rs]
    cats = defaultdict(list)
    for x in rs:
        for c,v in x["per_category"].items():
            cats[c].append(v["tokens_per_second"])
    return {
        "n": len(rs),
        "overall_mean": mean(overall),
        "overall_std": statistics.stdev(overall) if len(overall) >= 2 else 0.0,
        "cat_mean": {c: mean(vs) for c,vs in cats.items()}
    }

B  = summarize("baseline_v13")
M1 = summarize("medusa1")
M2 = summarize("medusa2")

print("\n=== OVERALL TOKENS/S (mean ± std) ===")
print(f"baseline_v13: n={B['n']:2d}  mean={B['overall_mean']:.2f}  std={B['overall_std']:.2f}")
print(f"medusa1     : n={M1['n']:2d}  mean={M1['overall_mean']:.2f}  std={M1['overall_std']:.2f}")
print(f"medusa2(v15): n={M2['n']:2d}  mean={M2['overall_mean']:.2f}  std={M2['overall_std']:.2f}")

def sp(a,b): 
    return a/b if b and b != 0 else float("nan")

print("\n=== SPEEDUP vs baseline_v13 ===")
print(f"medusa1/baseline_v13: {sp(M1['overall_mean'], B['overall_mean']):.3f}x  (fair)")
print(f"medusa2/baseline_v13: {sp(M2['overall_mean'], B['overall_mean']):.3f}x  (exploratory, cross-version)")

def per_cat_speedup(M, B):
    cats = sorted(set(M) | set(B))
    out = {}
    for c in cats:
        out[c] = sp(M.get(c, float('nan')), B.get(c, float('nan')))
    return out

pc1 = per_cat_speedup(M1["cat_mean"], B["cat_mean"])
pc2 = per_cat_speedup(M2["cat_mean"], B["cat_mean"])

print("\n=== PER-CATEGORY SPEEDUP: medusa1/baseline_v13 (fair) ===")
for c,v in pc1.items():
    print(f"{c:12s}: {v:.3f}x")

print("\n=== PER-CATEGORY SPEEDUP: medusa2/baseline_v13 (exploratory) ===")
for c,v in pc2.items():
    print(f"{c:12s}: {v:.3f}x")
PY

echo ""
echo "DONE."
echo "OUTDIR: $OUTDIR"
echo "CSV:    $CSV"
echo "JSON:   $JSONDIR"
echo "LOGS:   $LOGDIR"
