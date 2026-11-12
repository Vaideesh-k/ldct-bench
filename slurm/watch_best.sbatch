#!/bin/bash
# CPU-only watcher next to the CT-Mamba training job: keeps best_from11.pth
# (lowest train_loss among epochs >= 11) and epoch_<N>.pth every 25 epochs.
# See slurm/watch_best.py. Restart after a stop:  sbatch slurm/watch_best.sbatch
#SBATCH --job-name=ctm_watch
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=7-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/watch_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/watch_%j.err

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
export OMP_NUM_THREADS=1
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"
python slurm/watch_best.py
