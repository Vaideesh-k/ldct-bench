#!/bin/bash
# The detection comparison: run the trained nnU-Net on the locked LiTS test patients as
# (1) normal dose (upper bound), (2) simulated quarter dose, (3) each model's denoised output,
# then score lesion-wise detection with scripts/detect_eval.py -> results/detect_<input>.json
# and results/detect_summary.csv. Short GPU inference (MIG slice).
#     sbatch --dependency=afterok:<nnunet job>,afterok:<predict jobs...> slurm/detect_all.sbatch [models...]
#SBATCH --job-name=detect_all
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=04:00:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/detect_all_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/detect_all_%j.err

set -euo pipefail
MODELS=${*:-ct_mamba ctformer corediff}
source ~/venvs/nnunet/bin/activate
cd ~/ldct-bench
export nnUNet_raw=$HOME/ldct-bench/data/nnunet/raw
export nnUNet_preprocessed=$HOME/ldct-bench/data/nnunet/preprocessed
export nnUNet_results=$HOME/ldct-bench/runs/nnunet
export nnUNet_n_proc_DA=4 OMP_NUM_THREADS=4
TRAINER=${TRAINER:-nnUNetTrainer_250epochs}
CONFIG=${CONFIG:-3d_fullres}
echo "host: $(hostname) | date: $(date) | inputs: full quarter $MODELS"
du -sh ~ | tail -1

run_one() {   # name, extra args for prep_nnunet.py
  local name=$1; shift
  local img=data/nnunet/test_$name
  local out=runs/nnunet/pred_test_$name
  if [ ! -f "results/detect_$name.json" ]; then
    rm -rf "$img" "$out"
    python scripts/prep_nnunet.py --lits data/lits --split configs/lits_split.json --part test --images-only --out "$img" "$@"
    nnUNetv2_predict -i "$img" -o "$out" -d 501 -c "$CONFIG" -f 0 -tr "$TRAINER" -npp 4 -nps 4
    python scripts/detect_eval.py --lits data/lits --pred-root "$out" --out "results/detect_$name.json"
    rm -rf "$img"            # the NIfTI copies are not needed afterwards
  fi
  echo "detect_$name done"
}

run_one full
run_one quarter --from-predictions data/lits --from-kind quarter
for m in $MODELS; do
  if [ -d "runs/$m/predictions/lits" ]; then
    run_one "$m" --from-predictions "runs/$m/predictions/lits"
  else
    echo "$m: no LiTS predictions in runs/$m/predictions/lits, skipped"
  fi
done

python - <<'PY'
import csv, glob, json, os
rows = []
for f in sorted(glob.glob("results/detect_*.json")):
    s = json.load(open(f))["summary"]
    rows.append({"input": os.path.basename(f)[7:-5], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
with open("results/detect_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
PY
du -sh ~ | tail -1
echo "DETECT_ALL_DONE $(date)"
