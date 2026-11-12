#!/bin/bash
# 50-epoch short run of one model/variant on a full A100 (Idea 2 comparison). NOT submitted
# automatically: the owner reviews first. Chain the four runs one after another:
#     J0=$(sbatch --parsable --job-name=s50_ctmamba  slurm/short50.sbatch ct_mamba        ct_mamba_50)
#     J1=$(sbatch --parsable --job-name=s50_hyb_low  --dependency=afterany:$J0 slurm/short50.sbatch ct_mamba_hybrid hybrid_low_50)
#     J2=$(sbatch --parsable --job-name=s50_hyb_small --dependency=afterany:$J1 slurm/short50.sbatch ct_mamba_hybrid hybrid_small_50 configs/variants/hybrid_small.yaml)
#     J3=$(sbatch --parsable --job-name=s50_hyb_par  --dependency=afterany:$J2 slurm/short50.sbatch ct_mamba_hybrid hybrid_parallel_50 configs/variants/hybrid_parallel.yaml)
# Each: train 50 epochs (resumable), predict L506 with best.pth, evaluate (per-slice scores).
# Expected ~14 h each for CT-Mamba-sized models (250 epochs took 2d 22h = 17 min/epoch;
# the attention variants may differ, see the hybrid test job's timing).
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=2-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/short50_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/short50_%x_%j.err

set -euo pipefail
MODEL=${1:?model name}
RUN=${2:?run name}
VARIANT=${3:-}        # optional extra config, e.g. configs/variants/hybrid_small.yaml
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | model: $MODEL | run: $RUN | variant: ${VARIANT:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv
bash scripts/run_all.sh "$MODEL" --run "$RUN" --config configs/short50.yaml $VARIANT
cp "runs/$RUN/train_log.csv" "results/${RUN}_train_log.csv"
echo "finished: $(date)"
