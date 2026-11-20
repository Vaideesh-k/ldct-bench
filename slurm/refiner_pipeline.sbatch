#!/bin/bash
# Final-week steps 2-5 in ONE GPU job (the account may run only one GPU job at a time), with every
# large temporary file on node-local scratch so the 50 GB home quota does not grow:
#   1. evaluate the finished control  (L506 + LiTS images, nnU-Net detection)
#   2. Stage A: frozen CT-Mamba on the 11,226 training slices -> $SCR/stage_a (int16, ~6 GB)
#   3. train the refiner (checks, smoke run, 30,000 iterations; resumable), then evaluate it
#   4. the same for the 2.5D refiner (noisy slices z-1, z, z+1 + Stage A)
# Only small files reach home: checkpoints, per-slice score CSVs, detection JSONs, nnU-Net masks.
#     sbatch slurm/refiner_pipeline.sbatch
#SBATCH --job-name=refiner_pipe
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=2-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/refiner_pipe_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/refiner_pipe_%j.err
set -euo pipefail
cd ~/ldct-bench
SCR=${TMPDIR:-/tmp}/ldct_${USER}_${SLURM_JOB_ID}
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
free_gb=$(df --output=avail -BG "$SCR" | tail -1 | tr -dc 0-9)
echo "host: $(hostname) | date: $(date) | scratch $SCR (${free_gb} GB free)"; du -sh ~ | tail -1
[ "$free_gb" -ge 30 ] || { echo "SCRATCH_STOP: only ${free_gb} GB free on the node"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

detect() {   # run name: nnU-Net on its LiTS predictions in $SCR/pred/<run>/lits
  conda deactivate
  PRED_ROOT="$SCR/pred" SETNAME=lits NNWORK="$SCR/nn" bash slurm/detect_all.sbatch "$1"
  source ~/miniforge3/etc/profile.d/conda.sh; conda activate ctmamba
  rm -rf "$SCR/pred/$1" "$SCR/nn"
  du -sh ~ | tail -1
}

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba

# ---- 1. control evaluation
if [ ! -f results/detect_ct_mamba_control.json ]; then
  [ -f runs/ct_mamba_control/best.pth ] || { echo "no control checkpoint"; exit 4; }
  python scripts/predict.py --model ct_mamba --run ct_mamba_control --sets mayo_L506 lits --pred-root "$SCR/pred/ct_mamba_control"
  python scripts/evaluate.py --model ct_mamba --run ct_mamba_control --sets mayo_L506 lits --pred-root "$SCR/pred/ct_mamba_control"
  detect ct_mamba_control
fi
echo "CONTROL_EVAL_DONE $(date)"

# ---- 2. Stage A on node scratch (re-use the 1,000 slices made on 19 Nov)
export LDCT_STAGE_A_ROOT="$SCR/stage_a"
for d in data/mayo/*/ct_mamba data/lits/*/ct_mamba; do
  [ -d "$d" ] || continue
  p=$(basename "$(dirname "$d")"); mkdir -p "$LDCT_STAGE_A_ROOT/$p"; cp -r "$d" "$LDCT_STAGE_A_ROOT/$p/"
done
python scripts/precompute_stage_a.py --sets lits_train mayo_train
du -sh "$LDCT_STAGE_A_ROOT" | tail -1
echo "STAGE_A_DONE $(date)"

# ---- 3./4. train then evaluate each refiner (Stage A for the TEST sets = the stored CT-Mamba
#            predictions, same weights; the live check re-runs CT-Mamba on 3 slices to prove it)
env -u LDCT_STAGE_A_ROOT python tests/check_refiner.py | tail -2
for MODE in refiner refiner25d; do
  RUN=ct_mamba_$MODE
  export LDCT_STAGE_A_ROOT="$SCR/stage_a"
  python scripts/train_tumour.py --mode $MODE --smoke
  rm -rf "runs/${RUN}_smoke"
  python scripts/train_tumour.py --mode $MODE --resume
  echo "TRAIN_DONE $MODE $(date)"
  unset LDCT_STAGE_A_ROOT
  if [ ! -f "results/detect_$RUN.json" ]; then
    python scripts/predict_refiner.py --run $RUN --sets mayo_L506 lits --live-check 3 --pred-root "$SCR/pred/$RUN"
    python scripts/evaluate.py --model ct_mamba --run $RUN --sets mayo_L506 lits --pred-root "$SCR/pred/$RUN"
    detect $RUN
  fi
  echo "EVAL_DONE $MODE $(date)"
done
rm -rf "$SCR/stage_a"
echo "REFINER_PIPELINE_DONE $(date)"
