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
# Extended test set (final-week step 6): LITS=data/lits_ext SPLIT=configs/lits_ext_split.json TAG=ext_
# PRED_ROOT=scratch_ext  (model predictions in $PRED_ROOT/<model>/$SETNAME, default lits_ext, deleted after scoring)
# NNWORK=<dir>  temporary nnU-Net input copies (default data/nnunet; node-local /tmp keeps home small)
LITS=${LITS:-data/lits}
SPLIT=${SPLIT:-configs/lits_split.json}
TAG=${TAG:-}
PRED_ROOT=${PRED_ROOT:-}
CONFIG=${CONFIG:-3d_fullres}
echo "host: $(hostname) | date: $(date) | inputs: full quarter $MODELS"
du -sh ~ | tail -1

run_one() {   # name, extra args for prep_nnunet.py
  local name=$1; shift
  local img=${NNWORK:-data/nnunet}/test_$TAG$name
  local out=runs/nnunet/pred_test_$TAG$name
  if [ ! -f "results/detect_$TAG$name.json" ]; then
    rm -rf "$img" "$out"
    python scripts/prep_nnunet.py --lits "$LITS" --split "$SPLIT" --part test --images-only --out "$img" "$@"
    nnUNetv2_predict -i "$img" -o "$out" -d 501 -c "$CONFIG" -f 0 -tr "$TRAINER" -npp 4 -nps 4
    python scripts/detect_eval.py --lits "$LITS" --pred-root "$out" --out "results/detect_$TAG$name.json"
    rm -rf "$img"            # the NIfTI copies are not needed afterwards
  fi
  echo "detect_$TAG$name done"
}

run_one full
run_one quarter --from-predictions "$LITS" --from-kind quarter
for m in $MODELS; do
  if [ -n "$PRED_ROOT" ]; then pd="$PRED_ROOT/$m/${SETNAME:-lits_ext}"; else pd="runs/$m/predictions/lits"; fi
  if [ -d "$pd" ]; then
    run_one "$m" --from-predictions "$pd"
  else
    echo "$m: no LiTS predictions in $pd, skipped"
  fi
done

TAG="$TAG" python - <<'PY'
import csv, glob, json, os
tag = os.environ["TAG"]
rows = []
for f in sorted(glob.glob(f"results/detect_{tag}*.json")):
    name = os.path.basename(f)[7 + len(tag):-5]
    if not tag and name.startswith("ext_"):
        continue
    s = json.load(open(f))["summary"]
    rows.append({"input": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})
with open(f"results/detect_{tag}summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
PY
du -sh ~ | tail -1
echo "DETECT_ALL_DONE $(date)"
