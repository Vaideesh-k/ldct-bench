#!/bin/bash
# Idea 1 fine-tuning runs of CT-Mamba (NOT submitted automatically; owner reviews first).
# Needs data/lits (train patients converted) and runs/ct_mamba/best.pth.
#     J0=$(sbatch --parsable --job-name=ft_control slurm/finetune.sbatch ct_mamba_ft_control configs/finetune_control.yaml)
#     J1=$(sbatch --parsable --job-name=ft_lesion --dependency=afterany:$J0 slurm/finetune.sbatch ct_mamba_ft_lesion configs/finetune_lesion.yaml)
# 20 epochs over Mayo (5,410 slices) + LiTS train (~41 patients): roughly 20 x 25 min = ~9 h each.
# Afterwards: scripts/detect_eval.py on the nnU-Net output of the denoised LiTS test patients.
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=1-12:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/finetune_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/finetune_%x_%j.err

set -euo pipefail
RUN=${1:?run name}
CFG=${2:?config yaml}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | run: $RUN | config: $CFG"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train.py --model ct_mamba --run "$RUN" --config "$CFG" --init runs/ct_mamba/best.pth --resume
python scripts/predict.py --model ct_mamba --run "$RUN" --config "$CFG"
python scripts/evaluate.py --model ct_mamba --run "$RUN" --config "$CFG"
cp "runs/$RUN/train_log.csv" "results/${RUN}_train_log.csv"
echo "finished: $(date)"
