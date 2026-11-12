#!/bin/bash
# CT-Mamba training on one full A100 (40 GB): train (resumes from
# runs/ct_mamba/latest.pth), then predict and evaluate.
# Submit again after the 7-day limit:  cd ~/ldct-bench && sbatch slurm/ctmamba_train.sbatch
#SBATCH --job-name=ctmamba
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64000
#SBATCH --time=7-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/ctmamba_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/ctmamba_%j.err

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi
python -c "import torch; print(\"torch\", torch.__version__, \"| GPU:\", torch.cuda.get_device_name(0))"

# Official batch (8 slices x 4 patches = 32 patches per step, no accumulation)
# is already the setting in configs/, so no override is needed.
bash scripts/run_all.sh ct_mamba
echo "finished: $(date)"
