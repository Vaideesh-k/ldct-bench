#!/bin/bash
# Final-week step 6 in ONE GPU job, everything large on node-local scratch (home quota untouched):
#   1. convert the 10 extended LiTS test patients (configs/lits_ext_split.json) -> $SCR/lits_ext,
#      read straight from the remote Kaggle archives, same simulation as part 1 (48.2 HU target)
#   2. score quarter dose; nnU-Net detection on normal + quarter dose
#   3. every model: predict -> score images -> nnU-Net detection, then delete its predictions
#      (CT-Mamba first: its output is Stage A for both refiners)
# Only small files reach home: scores_lits_ext.csv, results/detect_ext_*.json, nnU-Net masks.
# Resumable: inputs whose results/detect_ext_<input>.json exists are skipped.
#     sbatch slurm/ext_pipeline.sbatch
#SBATCH --job-name=ext_pipe
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_2g.10gb:1   # every model peaks <= 1.7 GiB (measured 21 Nov); 4g slices were all busy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=2-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/ext_pipe_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/ext_pipe_%j.err
set -euo pipefail
cd ~/ldct-bench
SCR=${TMPDIR:-/tmp}/ldct_${USER}_${SLURM_JOB_ID}
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
free_gb=$(df --output=avail -BG "$SCR" | tail -1 | tr -dc 0-9)
echo "host: $(hostname) | date: $(date) | scratch $SCR (${free_gb} GB free)"; du -sh ~ | tail -1
[ "$free_gb" -ge 30 ] || { echo "SCRATCH_STOP: only ${free_gb} GB free on the node"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LDCT_LITS_EXT_ROOT="$SCR/lits_ext"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba

# ---- 1. conversion (disk guard measures the scratch, not home)
TARGET=$(python -c "import json; print(json.load(open('configs/lits_sim.json'))['target_added_noise_std_hu'])")
IDS=$(python -c "import json; print(' '.join(map(str, json.load(open('configs/lits_ext_split.json'))['test'])))")
echo "patients: $IDS | target noise $TARGET HU"
python scripts/prep_lits.py --remote --order $IDS --I0 9.167e5 --target-noise "$TARGET" --workers 16 \
    --out "$SCR/lits_ext" --raw "$SCR/raw" --home "$SCR" --max-used-gb 150
rm -rf "$SCR/raw"
mkdir -p results/lits_ext_meta
for d in "$SCR"/lits_ext/lits_*; do cp "$d/meta.json" "results/lits_ext_meta/$(basename "$d").json"; done
du -sh "$SCR/lits_ext" | tail -1; echo "EXT_PREP_DONE $(date)"

detect() {   # run name ('' = only normal + quarter dose)
  conda deactivate
  LITS="$SCR/lits_ext" SPLIT=configs/lits_ext_split.json TAG=ext_ PRED_ROOT="$SCR/pred" SETNAME=lits_ext \
    NNWORK="$SCR/nn" bash slurm/detect_all.sbatch $1
  source ~/miniforge3/etc/profile.d/conda.sh; conda activate ctmamba
  rm -rf "$SCR/nn"; du -sh ~ | tail -1
}

# ---- 2. quarter dose image scores
python scripts/evaluate.py --baseline --sets lits_ext

# ---- 3. models
one() {   # model run [stage-a]
  if [ ! -f "results/detect_ext_$2.json" ]; then
    if [ -n "${3:-}" ]; then
      python scripts/predict_refiner.py --run "$2" --sets lits_ext --pred-root "$SCR/pred/$2" --stage-a-root "$SCR/pred/ct_mamba" --live-check 3
    else
      python scripts/predict.py --model "$1" --run "$2" --sets lits_ext --pred-root "$SCR/pred/$2"
    fi
    python scripts/evaluate.py --model "$1" --run "$2" --sets lits_ext --pred-root "$SCR/pred/$2"
    detect "$2"
  fi
  echo "EXT_DONE $2 $(date)"
}
need_stage_a() { [ -f results/detect_ext_ct_mamba_refiner.json ] && [ -f results/detect_ext_ct_mamba_refiner25d.json ] || \
  [ -d "$SCR/pred/ct_mamba/lits_ext" ] || python scripts/predict.py --model ct_mamba --sets lits_ext --pred-root "$SCR/pred/ct_mamba"; }
one ct_mamba ct_mamba
need_stage_a
one ct_mamba ct_mamba_refiner stage_a;    rm -rf "$SCR/pred/ct_mamba_refiner"
one ct_mamba ct_mamba_refiner25d stage_a; rm -rf "$SCR/pred/ct_mamba_refiner25d" "$SCR/pred/ct_mamba"
one ct_mamba ct_mamba_control;            rm -rf "$SCR/pred/ct_mamba_control"
one ctformer ctformer;                    rm -rf "$SCR/pred/ctformer"
one corediff corediff;                    rm -rf "$SCR/pred/corediff"
du -sh ~ | tail -1; echo "EXT_PIPELINE_DONE $(date)"
