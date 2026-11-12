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
  echo "== calibration on train patient 0 (target $TARGET HU): iterate I0 <- I0 * (std/target)^2 until within 3 %"
  python - "$TARGET" "$ZIP" <<'PY'
import json, os, shutil, subprocess, sys
target, zip_path = float(sys.argv[1]), sys.argv[2]
I0, history = 2e5, []
for it in range(5):
    shutil.rmtree("data/lits_trial", ignore_errors=True)
    subprocess.run([sys.executable, "scripts/prep_lits.py", "--zip", zip_path, "--order", "0", "--I0", str(I0),
                    "--out", "data/lits_trial", "--raw", "data/lits_raw_tmp", "--workers", "16"], check=True)
    std = json.load(open("data/lits_trial/lits_000/meta.json"))["liver_added_noise_std_hu"]["median"]
    history.append({"I0": I0, "added_noise_std_hu": std})
    print(f"calibration round {it + 1}: I0 {I0:.4g} -> {std:.2f} HU (target {target:.2f})", flush=True)
    if abs(std - target) / target < 0.03:
        break
    I0 = I0 * (std / target) ** 2
out = {"I0": round(history[-1]["I0"], -2), "dose": 0.25, "sigma_e": 10.0, "target_added_noise_std_hu": target,
       "achieved_added_noise_std_hu": history[-1]["added_noise_std_hu"], "history": history,
       "calibration_patient": "lits_000 (train patient)",
       "rule": "I0 <- I0 * (std/target)^2 iterated on the trial patient until within 3 %; fixed for every patient"}
json.dump(out, open("configs/lits_sim.json", "w"), indent=1)
print("calibrated:", json.dumps(out), flush=True)
PY
  rm -rf data/lits_trial
fi
I0=$(python -c "import json; print(json.load(open('configs/lits_sim.json'))['I0'])")
echo "== converting all part-1 patients with I0 = $I0 (test patients first)"
python scripts/prep_lits.py --zip "$ZIP" --I0 "$I0" --workers 16 --raw data/lits_raw_tmp --order 0 $TEST $TRAIN
echo "== check: added noise in the first test patients vs target $TARGET HU"
python scripts/calibrate_lits_noise.py --lits data/lits/lits_002 data/lits/lits_004 data/lits/lits_000 --json results/lits_added_noise.json || true
