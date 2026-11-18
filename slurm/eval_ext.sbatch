#!/bin/bash
# Step 6: everything on the EXTENDED LiTS test set (data/lits_ext): predict with each model into
# scratch_ext/<model>/lits_ext (temporary, outside runs/), score images, nnU-Net detection
# (results/detect_ext_<input>.json), then delete that model's temporary predictions (regenerable;
# only one or two models' predictions are on disk at any time). Resumable: done inputs are skipped.
# CT-Mamba first: its output is the refiner's Stage A.
#     sbatch --dependency=afterany:<eval_refiner>,afterok:<ext_prep> slurm/eval_ext.sbatch
#SBATCH --job-name=eval_ext
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/eval_ext_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/eval_ext_%j.err
set -euo pipefail
cd ~/ldct-bench
SCR=scratch_ext
echo "host: $(hostname) | date: $(date)"; du -sh ~ | tail -1
export LITS=data/lits_ext SPLIT=configs/lits_ext_split.json TAG=ext_ PRED_ROOT=$SCR
source ~/miniforge3/etc/profile.d/conda.sh
one() {   # model run: predict -> score -> detect (only this run's temporary predictions on disk)
  if [ ! -f "results/detect_ext_$2.json" ]; then
    conda activate ctmamba
    if [ "$2" = ct_mamba_refiner ]; then
      [ -d "$SCR/ct_mamba/lits_ext" ] || python scripts/predict.py --model ct_mamba --sets lits_ext --pred-root "$SCR/ct_mamba"   # Stage A (resumed job)
      python scripts/predict_refiner.py --run "$2" --sets lits_ext --pred-root "$SCR/$2" --stage-a-root "$SCR/ct_mamba"
    else
      python scripts/predict.py --model "$1" --run "$2" --sets lits_ext --pred-root "$SCR/$2"
    fi
    python scripts/evaluate.py --model "$1" --run "$2" --sets lits_ext --pred-root "$SCR/$2"
    conda deactivate
    bash slurm/detect_all.sbatch "$2"
  fi
  du -sh ~ | tail -1
}
conda activate ctmamba; python scripts/evaluate.py --baseline --sets lits_ext; conda deactivate
one ct_mamba ct_mamba                       # kept until the refiner has used it as Stage A
one ct_mamba ct_mamba_refiner
rm -rf "$SCR/ct_mamba" "$SCR/ct_mamba_refiner"
one ct_mamba ct_mamba_control;  rm -rf "$SCR/ct_mamba_control"
one ctformer ctformer;          rm -rf "$SCR/ctformer"
one corediff corediff;          rm -rf "$SCR/corediff"
du -sh ~ | tail -1; echo "EVAL_EXT_DONE $(date)"
