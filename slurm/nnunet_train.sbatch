#!/bin/bash
# Tumour finder: nnU-Net v2 on the normal-dose LiTS train patients, one fold.
#     sbatch [--dependency=afterany:<jobid>] slurm/nnunet_train.sbatch [TRAINER] [CONFIG]
# Defaults: nnUNetTrainer_250epochs (the 1000-epoch default would take ~3-4 days), 3d_fullres.
# Steps: (1) write the raw dataset from data/lits (train part of the split only),
# (2) plan + preprocess, (3) train fold 0, (4) predict the locked test patients' NORMAL-dose
# images (upper bound for the detection metrics). Preprocessed data lives under
# data/nnunet (git-ignored); disk use is printed before and after.
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96000
#SBATCH --time=3-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/nnunet_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/nnunet_%j.err

set -euo pipefail
TRAINER=${1:-nnUNetTrainer_250epochs}
CONFIG=${2:-3d_fullres}
source ~/venvs/nnunet/bin/activate
cd ~/ldct-bench
export nnUNet_raw=$HOME/ldct-bench/data/nnunet/raw
export nnUNet_preprocessed=$HOME/ldct-bench/data/nnunet/preprocessed
export nnUNet_results=$HOME/ldct-bench/runs/nnunet
export nnUNet_n_proc_DA=8 OMP_NUM_THREADS=4
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
echo "host: $(hostname) | date: $(date) | trainer: $TRAINER | config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv
du -sh ~ | tail -1

echo "== 1/4 raw dataset"
python scripts/prep_nnunet.py --lits data/lits --split configs/lits_split.json --out "$nnUNet_raw" --dataset-id 501
echo "== 2/4 plan + preprocess"
nnUNetv2_plan_and_preprocess -d 501 -c "$CONFIG" -np 8 --verify_dataset_integrity
du -sh data/nnunet | tail -1
echo "== 3/4 train fold 0"
nnUNetv2_train 501 "$CONFIG" 0 -tr "$TRAINER" --npz || nnUNetv2_train 501 "$CONFIG" 0 -tr "$TRAINER" --npz --c
echo "== 4/4 predict the locked test patients (normal dose)"
python scripts/prep_nnunet.py --lits data/lits --split configs/lits_split.json --part test --images-only --out data/nnunet/test_full
nnUNetv2_predict -i data/nnunet/test_full -o runs/nnunet/pred_test_full -d 501 -c "$CONFIG" -f 0 -tr "$TRAINER"
python scripts/detect_eval.py --lits data/lits --pred-root runs/nnunet/pred_test_full --out results/detect_nnunet_fulldose.json || true
du -sh ~ | tail -1
echo "NNUNET_DONE $(date)"
