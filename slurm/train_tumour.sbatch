#!/bin/bash
# Final-week steps 3/4: train the refiner or the control (scripts/train_tumour.py).
#     sbatch --job-name=control --gres=gpu:a100_4g.20gb:1 slurm/train_tumour.sbatch control
#     sbatch --job-name=refiner --gres=gpu:a100_4g.20gb:1 slurm/train_tumour.sbatch refiner
# Runs the CPU checks, a 40-iteration smoke run, then the real run (resumable: resubmit).
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=3-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/tumour_%x_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/tumour_%x_%j.err
set -euo pipefail
MODE=${1:?refiner or control}; shift
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | mode: $MODE | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
du -sh ~ | tail -1
python tests/check_refiner.py | tail -2
python - <<'PY'
import torch
a = torch.load("runs/ct_mamba/best.pth", map_location="cpu"); b = torch.load("runs/ct_mamba/best_from11.pth", map_location="cpu")
same = a["model"].keys() == b["model"].keys() and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
print(f"best.pth epoch {a.get('epoch')} / best_from11.pth epoch {b.get('epoch')}: identical weights = {same}")
PY
python scripts/train_tumour.py --mode "$MODE" --smoke "$@"
rm -rf "runs/ct_mamba_${MODE}_smoke" "runs/ct_mamba_refiner_smoke" "runs/ct_mamba_control_smoke"
python scripts/train_tumour.py --mode "$MODE" --resume "$@"
du -sh ~ | tail -1
echo "TUMOUR_TRAIN_DONE $MODE $(date)"
