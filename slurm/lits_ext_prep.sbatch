#!/bin/bash
# Step 6: convert the extended LiTS test patients (configs/lits_ext_split.json) into data/lits_ext,
# one patient at a time, read straight from the remote Kaggle archives (no archive stored);
# same simulation as part 1 (per-patient exposure calibrated to 48.2 HU added liver noise).
# Resumable; stops before the home folder passes 39 GB (keep >= 10 GB free).
#     sbatch slurm/lits_ext_prep.sbatch
#SBATCH --job-name=ext_prep
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000
#SBATCH --time=2-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/ext_prep_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/ext_prep_%j.err
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"; du -sh ~ | tail -1
TARGET=$(python -c "import json; print(json.load(open('configs/lits_sim.json'))['target_added_noise_std_hu'])")
IDS=$(python -c "import json; print(' '.join(map(str, json.load(open('configs/lits_ext_split.json'))['test'])))")
echo "patients: $IDS | target noise $TARGET HU"
python scripts/prep_lits.py --remote --order $IDS --I0 9.167e5 --target-noise "$TARGET" --workers 16 \
    --out data/lits_ext --raw data/lits_raw_tmp --max-used-gb 39.0
du -sh ~ | tail -1; echo "EXT_PREP_DONE $(date)"
