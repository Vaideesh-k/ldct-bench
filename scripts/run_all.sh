#!/usr/bin/env bash
# Train (or continue training) one model, then predict and evaluate it on all test sets.
#
#     bash scripts/run_all.sh ct_mamba
#
# Safe to run again after a crash: training continues from runs/<model>/latest.pth,
# and prediction/evaluation simply run again. Extra arguments (e.g. --config file.yaml)
# are passed to every step. Uses $PYTHON if set, else "python" from the active environment.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/run_all.sh <model> [--config extra.yaml]"
  exit 1
fi
MODEL=$1
shift
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."   # run from the repo root, wherever we were called from

echo "== 1/4 train $MODEL (continues if a run exists) =="
$PY scripts/train.py --model "$MODEL" --resume "$@"

echo "== 2/4 predict all test sets with best.pth =="
$PY scripts/predict.py --model "$MODEL" "$@"

echo "== 3/4 evaluate $MODEL =="
$PY scripts/evaluate.py --model "$MODEL" "$@"

echo "== 4/4 evaluate the quarter-dose input (baseline row for comparison) =="
$PY scripts/evaluate.py --baseline "$@"

echo "Done. Results: results/summary.csv"
