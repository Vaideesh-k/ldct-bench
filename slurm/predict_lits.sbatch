#!/bin/bash
# Predict + evaluate one already-trained model on the LiTS test set (locked patients) on a MIG slice.
#     sbatch --job-name=lits_ctmamba slurm/predict_lits.sbatch ct_mamba
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/predlits_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/predlits_%x_%j.err

set -euo pipefail
MODEL=${1:?model}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | model: $MODEL"
python scripts/predict.py --model "$MODEL" --sets lits
python scripts/evaluate.py --model "$MODEL" --sets lits
echo "PREDLITS_OK $(date)"
