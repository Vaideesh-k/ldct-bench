#!/bin/bash
# Step 7: the zoomed small-tumour figure (small_tumour.png).
#   1. pick the tumour by the fixed rule (largest < 10 mm in the LiTS locked test set; masks only)
#   2. predict ONLY that patient (LDCT_ONLY_PATIENTS) with control, 2D and 2.5D refiner -> node scratch
#      (CT-Mamba / CTformer / CoreDiff predictions are already in runs/<model>/predictions/lits)
#   3. draw the figure; only the PNG reaches home
#     sbatch slurm/figures.sbatch
#SBATCH --job-name=figures
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=04:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/figures_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/figures_%j.err
set -euo pipefail
cd ~/ldct-bench
SCR=${TMPDIR:-/tmp}/ldct_${USER}_${SLURM_JOB_ID}
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT
echo "host: $(hostname) | date: $(date) | scratch $SCR"; du -sh ~ | tail -1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba

PID=$(python scripts/make_figures.py --part pick)
echo "chosen patient: $PID"
export LDCT_ONLY_PATIENTS="$PID"
python scripts/predict.py --model ct_mamba --run ct_mamba_control --sets lits --pred-root "$SCR/pred/ct_mamba_control"
for RUN in ct_mamba_refiner ct_mamba_refiner25d; do
    python scripts/predict_refiner.py --run $RUN --sets lits --live-check 3 --pred-root "$SCR/pred/$RUN"
done
python scripts/make_figures.py --part zoom --pred-root "$SCR/pred" --out results/figures
du -sh ~ | tail -1
echo "done: $(date)"
