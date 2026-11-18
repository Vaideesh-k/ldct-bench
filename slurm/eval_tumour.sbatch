#!/bin/bash
# Step 5 for one new model: predict Mayo L506 + the LiTS locked test set, score images, then
# nnU-Net detection on its LiTS output (same nnU-Net, same scorer as the baselines).
#     sbatch --dependency=afterok:<train job> slurm/eval_tumour.sbatch control
#     sbatch --dependency=afterok:<train job> slurm/eval_tumour.sbatch refiner
#SBATCH --job-name=eval_tumour
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=08:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/eval_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/eval_%x_%j.err
set -euo pipefail
MODE=${1:?refiner or control}
RUN=ct_mamba_$MODE
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | $RUN"; du -sh ~ | tail -1
# predictions need ~2.1 GB (float32, 1,984 slices); keep 10 GB of the 50 GB quota free
used=$(du -s --block-size=1M ~ | cut -f1)
if [ $((used + 2300)) -gt $((40 * 1024)) ]; then
  echo "DISK_STOP: home uses $((used / 1024)) GB; predictions would leave < 10 GB free. Free space, then resubmit."; exit 3
fi
[ -f "runs/$RUN/best.pth" ] || { echo "no runs/$RUN/best.pth: training not done"; exit 4; }
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
if [ "$MODE" = refiner ]; then
  python scripts/predict_refiner.py --run "$RUN" --sets mayo_L506 lits --live-check 3
else
  python scripts/predict.py --model ct_mamba --run "$RUN" --sets mayo_L506 lits
fi
python scripts/evaluate.py --model ct_mamba --run "$RUN" --sets mayo_L506 lits
du -sh ~ | tail -1
conda deactivate
bash slurm/detect_all.sbatch "$RUN"
echo "EVAL_TUMOUR_DONE $RUN $(date)"
