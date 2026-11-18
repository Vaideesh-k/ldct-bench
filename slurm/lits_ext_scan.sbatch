#!/bin/bash
# Step 6: count tumours of LiTS patients 51-130 from their masks only (no images, no models).
#     sbatch slurm/lits_ext_scan.sbatch
#SBATCH --job-name=ext_scan
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/ext_scan_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/ext_scan_%j.err
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"; du -sh ~ | tail -1
python scripts/scan_lits_ext.py --patients 51-130 --out results/lits_ext_scan.json --tmp data/lits_raw_tmp
du -sh ~ | tail -1; echo "EXT_SCAN_DONE $(date)"
