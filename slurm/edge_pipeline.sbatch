#!/bin/bash
# Edge-preserving fine-tune of CT-Mamba (docs/success_criteria_edge.md) in ONE GPU job, everything
# large on node-local scratch (home is at ~41 GB of 50; only small files and an fp16 best.pth reach it):
#   0. smoke test of the edge loss (40 iterations)
#   1. convert the fresh test set (configs/lits_fresh_split.json) -> $SCR/lits_fresh
#   2. every existing input on lits_fresh: image scores + nnU-Net detection
#   3. train ct_mamba_edge (control setup + gradient term), 30,000 iterations
#   4. edge model: Mayo L506 + lits_fresh (primary), then lits and lits_ext (secondary)
# Resumable for the scoring parts: inputs whose results/detect_<tag><input>.json exists are skipped.
#     sbatch slurm/edge_pipeline.sbatch
#SBATCH --job-name=edge_pipe
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=4-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/edge_pipe_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/edge_pipe_%j.err
set -euo pipefail
cd ~/ldct-bench
SCR=${TMPDIR:-/tmp}/ldct_${USER}_${SLURM_JOB_ID}
mkdir -p "$SCR"
cleanup() { for l in runs/ct_mamba_edge runs/ct_mamba_edge_smoke; do [ -L "$l" ] && rm "$l"; done; rm -rf "$SCR"; }
trap cleanup EXIT
free_gb=$(df --output=avail -BG "$SCR" | tail -1 | tr -dc 0-9)
echo "host: $(hostname) | date: $(date) | scratch $SCR (${free_gb} GB free)"; du -sh ~ | tail -1
[ "$free_gb" -ge 60 ] || { echo "SCRATCH_STOP: only ${free_gb} GB free on the node"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LDCT_LITS_FRESH_ROOT="$SCR/lits_fresh" LDCT_LITS_EXT_ROOT="$SCR/lits_ext"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba

# training runs live on scratch: runs/<run> is a symlink there for the job's lifetime
[ -e runs/ct_mamba_edge ] && { echo "runs/ct_mamba_edge exists; not overwriting"; exit 4; }
mkdir -p "$SCR/runs/edge" "$SCR/runs/smoke"
ln -s "$SCR/runs/edge" runs/ct_mamba_edge
ln -s "$SCR/runs/smoke" runs/ct_mamba_edge_smoke

# ---- 0. smoke
python scripts/train_tumour.py --mode edge --run ct_mamba_edge --smoke
echo "SMOKE_DONE $(date)"

# ---- 1. fresh test set
TARGET=$(python -c "import json; print(json.load(open('configs/lits_sim.json'))['target_added_noise_std_hu'])")
convert() {   # split out
  local ids; ids=$(python -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['test'])))" "$1")
  echo "convert $1: $ids"
  python scripts/prep_lits.py --remote --order $ids --I0 9.167e5 --target-noise "$TARGET" --workers 16 \
      --out "$2" --raw "$SCR/raw" --home "$SCR" --max-used-gb 150
  rm -rf "$SCR/raw"
}
convert configs/lits_fresh_split.json "$SCR/lits_fresh"
mkdir -p results/lits_fresh_meta
for d in "$SCR"/lits_fresh/lits_*; do cp "$d/meta.json" "results/lits_fresh_meta/$(basename "$d").json"; done
du -sh "$SCR/lits_fresh" | tail -1; echo "FRESH_PREP_DONE $(date)"

detect() {   # set tag split lits-root run ('' = normal + quarter dose only)
  conda deactivate
  LITS="$4" SPLIT="$3" TAG="$2" PRED_ROOT="$SCR/pred" SETNAME="$1" NNWORK="$SCR/nn" bash slurm/detect_all.sbatch $5
  source ~/miniforge3/etc/profile.d/conda.sh; conda activate ctmamba
  rm -rf "$SCR/nn"; du -sh ~ | tail -1
}
one() {   # set tag split lits-root model run [stage_a]
  if [ ! -f "results/detect_$2$6.json" ]; then
    if [ -n "${7:-}" ]; then
      python scripts/predict_refiner.py --run "$6" --sets "$1" --pred-root "$SCR/pred/$6" --stage-a-root "$SCR/pred/ct_mamba" --live-check 3
    elif [ ! -d "$SCR/pred/$6/$1" ]; then
      python scripts/predict.py --model "$5" --run "$6" --sets "$1" --pred-root "$SCR/pred/$6"
    fi
    python scripts/evaluate.py --model "$5" --run "$6" --sets "$1" --pred-root "$SCR/pred/$6"
    detect "$1" "$2" "$3" "$4" "$6"
  fi
  echo "DONE $1 $6 $(date)"
}
F=(lits_fresh fresh_ configs/lits_fresh_split.json "$SCR/lits_fresh")

# ---- 2. existing inputs on the fresh set
python scripts/evaluate.py --baseline --sets lits_fresh
[ -f results/detect_fresh_quarter.json ] || detect "${F[@]}" ""
[ -f results/detect_fresh_ct_mamba_refiner25d.json ] || \
  python scripts/predict.py --model ct_mamba --sets lits_fresh --pred-root "$SCR/pred/ct_mamba"   # also Stage A
one "${F[@]}" ct_mamba ct_mamba
one "${F[@]}" ct_mamba ct_mamba_refiner stage_a;    rm -rf "$SCR/pred/ct_mamba_refiner"
one "${F[@]}" ct_mamba ct_mamba_refiner25d stage_a; rm -rf "$SCR/pred/ct_mamba_refiner25d" "$SCR/pred/ct_mamba"
one "${F[@]}" ct_mamba ct_mamba_control;            rm -rf "$SCR/pred/ct_mamba_control"
one "${F[@]}" ctformer ctformer;                    rm -rf "$SCR/pred/ctformer"
one "${F[@]}" corediff corediff;                    rm -rf "$SCR/pred/corediff"
echo "BASELINES_DONE $(date)"

# ---- 3. train the edge model
python scripts/train_tumour.py --mode edge --run ct_mamba_edge
echo "TRAIN_DONE $(date)"
mkdir -p results/ct_mamba_edge
cp runs/ct_mamba_edge/train_log.csv results/ct_mamba_edge/train_log.csv

# ---- 4. edge model: primary (L506, lits_fresh), then secondary (lits, lits_ext)
python scripts/predict.py --model ct_mamba --run ct_mamba_edge --sets mayo_L506 --pred-root "$SCR/pred/edge_mayo"
python scripts/evaluate.py --model ct_mamba --run ct_mamba_edge --sets mayo_L506 --pred-root "$SCR/pred/edge_mayo"
rm -rf "$SCR/pred/edge_mayo"
one "${F[@]}" ct_mamba ct_mamba_edge; rm -rf "$SCR/pred/ct_mamba_edge"
cp runs/ct_mamba_edge/*.json runs/ct_mamba_edge/*.csv results/ct_mamba_edge/ 2>/dev/null || true
echo "PRIMARY_DONE $(date)"

one lits "" configs/lits_split.json data/lits ct_mamba ct_mamba_edge; rm -rf "$SCR/pred/ct_mamba_edge"
convert configs/lits_ext_split.json "$SCR/lits_ext"
one lits_ext ext_ configs/lits_ext_split.json "$SCR/lits_ext" ct_mamba ct_mamba_edge; rm -rf "$SCR/pred/ct_mamba_edge"
cp runs/ct_mamba_edge/*.json runs/ct_mamba_edge/*.csv results/ct_mamba_edge/ 2>/dev/null || true

# keep the weights (fp16, ~150 MB) and the small files at home; the scratch run goes with the job
python - <<'PY'
import torch
ck = torch.load("runs/ct_mamba_edge/best.pth", map_location="cpu")
ck["model"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
torch.save(ck, "results/ct_mamba_edge/best_fp16.pth")
PY
du -sh ~ | tail -1; echo "EDGE_PIPELINE_DONE $(date)"
