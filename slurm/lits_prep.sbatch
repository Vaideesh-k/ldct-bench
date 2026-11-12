#!/bin/bash
# Build data/lits (download + convert + simulate, one patient at a time) on the CPU partition.
#     sbatch slurm/lits_prep.sbatch <I0> <patient numbers in order...>
#SBATCH --job-name=lits_prep
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000
#SBATCH --time=1-12:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/lits_prep_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/lits_prep_%j.err

set -euo pipefail
I0=${1:?I0}
shift
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date) | I0 $I0 | order: $*"
du -sh ~ | tail -1
python scripts/prep_lits.py --I0 "$I0" --workers 16 --order "$@"
du -sh ~ | tail -1
echo "LITS_PREP_DONE $(date)"
