#!/usr/bin/env bash
# Unattended LiTS build: calibrate the simulated dose on TRAIN patient 0, then convert
# everyone (locked test patients first, then the train patients) from the part-1 archive.
#     bash scripts/lits_pipeline.sh            (run by slurm/lits_prep.sbatch)
# Target: Mayo median std(quarter - full) in uniform liver-like tissue (results/mayo_added_noise.json).
# Noise std scales with 1/sqrt(I0), so  I0 = I0_trial * (std_trial / std_target)^2.
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP=data/lits_raw/liver-tumor-segmentation.zip
TARGET=$(python -c "import json; print(json.load(open('results/mayo_added_noise.json'))['mayo']['added_noise_std_median_hu'])")
TEST="2 4 10 11 22 26 28 29 32 45"
TRAIN=$(python -c "import json; print(' '.join(str(k) for k in json.load(open('configs/lits_split.json'))['train']))")

if [ ! -f configs/lits_sim.json ]; then
  echo "== calibration: trial conversion of train patient 0 at I0 = 2e5 (target $TARGET HU)"
  rm -rf data/lits_trial data/lits_raw_trial
  python scripts/prep_lits.py --zip "$ZIP" --order 0 --I0 2e5 --out data/lits_trial --raw data/lits_raw_trial --workers 16
  python - "$TARGET" <<'PY'
import json, sys
target = float(sys.argv[1])
meta = json.load(open("data/lits_trial/lits_000/meta.json"))
trial = meta["liver_added_noise_std_hu"]["median"]
I0 = 2e5 * (trial / target) ** 2
out = {"I0": round(I0, -2), "dose": 0.25, "target_added_noise_std_hu": target, "trial_I0": 2e5,
       "trial_added_noise_std_hu": trial, "calibration_patient": "lits_000 (train patient)",
       "rule": "I0 = trial_I0 * (trial_std / target_std)^2; fixed for every patient"}
json.dump(out, open("configs/lits_sim.json", "w"), indent=1)
print("calibrated:", out)
PY
  rm -rf data/lits_trial data/lits_raw_trial
fi
I0=$(python -c "import json; print(json.load(open('configs/lits_sim.json'))['I0'])")
echo "== converting all part-1 patients with I0 = $I0 (test patients first)"
python scripts/prep_lits.py --zip "$ZIP" --I0 "$I0" --workers 16 --order 0 $TEST $TRAIN
echo "== check: added noise in the first test patients vs target $TARGET HU"
python scripts/calibrate_lits_noise.py --lits data/lits/lits_002 data/lits/lits_004 data/lits/lits_000 --json results/lits_added_noise.json || true
