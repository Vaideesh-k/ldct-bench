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

# Per-patient exposure calibration (AEC-like): every patient gets the I0 that produces
# TARGET HU of added liver noise. configs/lits_sim.json records the rule.
python - "$TARGET" <<'PY'
import json, sys
json.dump({"mode": "per-patient exposure calibration", "target_added_noise_std_hu": float(sys.argv[1]),
           "dose": 0.25, "sigma_e": 10.0, "I0_start": 9.167e5,
           "rule": "for each patient iterate I0 <- I0 * (std/target)^2 on up to 12 liver slices until within 3 %, "
                   "then simulate all slices; the I0 and its history are in each patient's meta.json",
           "why": "with one fixed I0 the added liver noise ranged 32-686 HU over the 51 patients (patient and "
                  "pixel size); real scanners modulate exposure, and the Mayo patients span only 37-62 HU"},
          open("configs/lits_sim.json", "w"), indent=1)
PY
echo "== converting all part-1 patients, per-patient calibration to $TARGET HU (test patients first)"
if [ "${QUARTER_ONLY:-0}" = "1" ]; then   # patients already converted: only the low-dose slices are redone, no archive needed
  python scripts/prep_lits.py --from-existing --quarter-only --I0 9.167e5 --target-noise "$TARGET" --workers 16 --order 0 $TEST $TRAIN
else
  python scripts/prep_lits.py --zip "$ZIP" --I0 9.167e5 --target-noise "$TARGET" --workers 16 --raw data/lits_raw_tmp --order 0 $TEST $TRAIN
fi
echo "== check: added noise vs target $TARGET HU"
python scripts/calibrate_lits_noise.py --lits data/lits/lits_* --json results/lits_added_noise.json | tail -55 || true
echo "== quarter-dose baseline row on the LiTS test set"
python scripts/evaluate.py --baseline --sets lits
