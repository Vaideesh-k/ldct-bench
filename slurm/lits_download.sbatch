#!/bin/bash
# Download the LiTS part-1 archive (5.2 GB) with the Kaggle CLI on a CPU node.
# Per-file downloads return 404 for this dataset and the CLI buffers the archive in RAM,
# so this runs on cpu_student (16 GB RAM), never on the login node.
#     sbatch slurm/lits_download.sbatch
#SBATCH --job-name=lits_dl
#SBATCH --partition=cpu_student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=06:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/lits_dl_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/lits_dl_%j.err

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
mkdir -p data/lits_raw
echo "host: $(hostname) | date: $(date)"
du -sh ~ | tail -1
kaggle datasets download -d andrewmvd/liver-tumor-segmentation -p data/lits_raw --force
ls -la data/lits_raw
python -c "import zipfile; z = zipfile.ZipFile('data/lits_raw/liver-tumor-segmentation.zip'); n = z.namelist(); print(len(n), 'members;', sum(1 for x in n if x.startswith('volume')), 'volumes;', sum(1 for x in n if x.startswith('segmentations')), 'masks'); print('testzip:', z.testzip())"
du -sh ~ | tail -1
echo "LITS_DL_DONE $(date)"
