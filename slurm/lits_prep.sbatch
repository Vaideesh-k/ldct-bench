#!/bin/bash
# Build data/lits (download + convert + simulate, one patient at a time) on the CPU partition.
#     sbatch [--dependency=afterok:<download job>] slurm/lits_prep.sbatch
# Runs scripts/lits_pipeline.sh: calibrate I0 on train patient 0, then convert everyone from
# the part-1 archive data/lits_raw/liver-tumor-segmentation.zip (see scripts/prep_lits.py).
#SBATCH --job-name=lits_prep
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000
#SBATCH --time=1-12:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/lits_prep_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/lits_prep_%j.err

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"
du -sh ~ | tail -1
bash scripts/lits_pipeline.sh
du -sh ~ | tail -1
echo "LITS_PREP_DONE $(date)"
