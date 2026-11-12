#!/bin/bash
# Smoke test + step timing of one model on a MIG slice (short, runs next to long jobs).
#     sbatch --job-name=smoke_ctformer slurm/smoke_model.sbatch ctformer
#     sbatch --job-name=smoke_corediff slurm/smoke_model.sbatch corediff 2   # 2 = batch override for the timing
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=00:50:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/smoke_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/smoke_%x_%j.err

set -euo pipefail
MODEL=${1:?model name}
BATCH=${2:-}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | model: $MODEL"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== tests"
python tests/check_baselines.py 2>&1 | tail -3
echo "=== smoke (1 epoch, 5 steps, in runs/${MODEL}_smoke)"
python scripts/train.py --model "$MODEL" --smoke
echo "=== timing"
python slurm/bench_step.py --model "$MODEL" ${BATCH:+--batch $BATCH}
echo "SMOKE_OK $(date)"
