#!/bin/bash
# Step 2: frozen CT-Mamba outputs for the refiner's training slices (int16, ~5 GB). Resumable.
#     sbatch [--dependency=afterany:<job>] slurm/precompute_stage_a.sbatch
#SBATCH --job-name=stage_a
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/stage_a_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/stage_a_%j.err
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"; du -sh ~ | tail -1
python scripts/precompute_stage_a.py --sets lits_train mayo_train --max-used-gb 39.7
du -sh ~ | tail -1; echo "STAGE_A_DONE $(date)"
