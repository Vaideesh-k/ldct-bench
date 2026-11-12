#!/bin/bash
# Full run of one model on one full A100: train (resumes from runs/<model>/latest.pth),
# then predict on every test set that exists and evaluate (per-slice scores included).
#     sbatch --job-name=ctformer slurm/train_model.sbatch ctformer
#     sbatch --job-name=corediff --dependency=afterany:<jobid> slurm/train_model.sbatch corediff
# If the 7-day limit hits first, submit the same command again: training resumes.
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=7-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/train_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/train_%x_%j.err

set -euo pipefail
MODEL=${1:?model name}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | model: $MODEL"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"
df -h ~ | tail -1

bash scripts/run_all.sh "$MODEL"
cp "runs/$MODEL/train_log.csv" "results/${MODEL}_train_log.csv"
echo "finished: $(date)"
