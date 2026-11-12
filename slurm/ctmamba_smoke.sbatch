#!/bin/bash
# CT-Mamba smoke test + step timing on one full A100 (before the 7-day run).
#SBATCH --job-name=ctm_smoke
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/smoke_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/smoke_%j.err

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== smoke"
python scripts/train.py --model ct_mamba --smoke
echo "=== timing"
python slurm/bench_step.py
echo "SMOKE_OK $(date)"
