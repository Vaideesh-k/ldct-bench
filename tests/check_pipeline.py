"""End-to-end check of the whole pipeline with RED-CNN on small fake data.

Runs the real scripts (train, predict, evaluate, run_all.sh) exactly as a user
would, on a Mac (mps), an NVIDIA GPU or the CPU, in a minute or two.
Everything is written to a temporary folder that is deleted at the end; the
real data/, runs/ and results/ folders are never touched (checked at the end).
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ldct.data import SlicePairs, list_pairs, unit_to_hu

PATIENTS = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"]
failed = []


def check(label, ok):
    """Print one result and remember failures (the script exits with code 1 at the end)."""
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


def run(args, expect_ok=True):
    """Run a command from the repo root; show its output if it fails unexpectedly."""
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if expect_ok and r.returncode != 0:
        print(f"--- {' '.join(args)} failed (exit {r.returncode}):\n{out[-3000:]}\n---")
    return r.returncode, out


def script(name, *args, expect_ok=True):
    return run([sys.executable, os.path.join("scripts", name), *args], expect_ok)


def fake_slice(rng, size=128):
    """A round 'body' of soft tissue (40 HU) with a 'bone' spot, in air (-1000 HU)."""
    y, x = np.mgrid[:size, :size]
    r = np.hypot(y - size / 2, x - size / 2)
    img = np.where(r < size * 0.4, 40.0, -1000.0)
    img[(r > 10) & (r < 16)] = 700.0
    return img + rng.normal(0, 5, img.shape)


def save_pair(root, patient, name, full, rng):
    """Save like prep_mayo.py does: int16 HU; quarter-dose = full-dose + noise."""
    quarter = full + rng.normal(0, 40, full.shape)
    for kind, img in (("full", full), ("quarter", quarter)):
        os.makedirs(os.path.join(root, patient, kind), exist_ok=True)
        np.save(os.path.join(root, patient, kind, f"{name}.npy"), np.round(img).astype(np.int16))


def snapshot(folder):
    """File names, sizes and change times, to prove the real folders were not touched."""
    found = {}
    for dirpath, _, files in os.walk(os.path.join(REPO, folder)):
        for f in files:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            found[p] = (st.st_size, st.st_mtime)
    return found


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


real_before = {d: snapshot(d) for d in ("data", "runs", "results")}
tmp = tempfile.mkdtemp(prefix="ldct_pipeline_")
try:
    # ---- fake data: 10 Mayo patients x 6 slices, 1 LiTS volume x 2 slices ----
    rng = np.random.default_rng(0)
    mayo, lits = os.path.join(tmp, "mayo"), os.path.join(tmp, "lits")
    for p in PATIENTS:
        for i in range(6):
            save_pair(mayo, p, f"{p}_{i:04d}", fake_slice(rng), rng)
    for i in range(2):
        save_pair(lits, "volume-0", f"volume-0_{i:04d}", fake_slice(rng), rng)

    runs, results = os.path.join(tmp, "runs"), os.path.join(tmp, "results")
    cfg = {
        "data": {"mayo_root": mayo,
                 "test_sets": {"mayo_L506": {"root": mayo, "patients": ["L506"]},
                               "lits": {"root": lits, "patients": "all"}}},
        "train": {"epochs": 2, "batch_size": 2, "patch_n": 2, "workers": 0,
                  "val_every": 1, "max_steps_per_epoch": 3},
        "output_dir": runs,
        "results_dir": results,
    }
    cfg_path = os.path.join(tmp, "test_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    C = ["--config", cfg_path]

    # ---- training ----
    code, out = script("train.py", "--model", "redcnn", "--smoke", *C)
    smoke = os.path.join(runs, "redcnn_smoke")
    check("smoke run finished", code == 0)
    check("smoke run saved latest.pth + best.pth + 1 log row",
          all(os.path.exists(os.path.join(smoke, f)) for f in ("latest.pth", "best.pth"))
          and len(read_csv(os.path.join(smoke, "train_log.csv"))) == 1)
    check("parameter count printed (1.85 M)", "1,848,865 parameters" in out)

    run_dir = os.path.join(runs, "redcnn")
    code, out = script("train.py", "--model", "redcnn", "--epochs", "2", *C)
    check("2 short epochs finished", code == 0 and len(read_csv(os.path.join(run_dir, "train_log.csv"))) == 2)
    check("smoke run kept apart from the real run", os.path.isdir(smoke) and os.path.isdir(run_dir))

    code, out = script("train.py", "--model", "redcnn", *C, expect_ok=False)
    check("refuses to overwrite a run without --resume", code != 0 and "--resume" in out)

    code, out = script("train.py", "--model", "redcnn", "--resume", "--epochs", "3", *C)
    log = read_csv(os.path.join(run_dir, "train_log.csv"))
    check("resume continued at epoch 3", code == 0 and "Resuming after epoch 2" in out
          and [r["epoch"] for r in log] == ["1", "2", "3"])
    check("log has epoch, train_loss, val_l1, lr, seconds",
          list(log[0]) == ["epoch", "train_loss", "val_l1", "lr", "seconds"] and all(r["val_l1"] for r in log))
    check("only latest.pth + best.pth (no file per epoch)",
          sorted(f for f in os.listdir(run_dir) if f.endswith(".pth")) == ["best.pth", "latest.pth"])
    best = torch.load(os.path.join(run_dir, "best.pth"), map_location="cpu")
    losses = {int(r["epoch"]): float(r["train_loss"]) for r in log}
    check("best.pth = epoch with the lowest training loss", best["epoch"] == min(losses, key=losses.get))

    # ---- prediction ----
    code, out = script("predict.py", "--model", "redcnn", *C)
    pred_root = os.path.join(run_dir, "predictions")
    mayo_preds = os.listdir(os.path.join(pred_root, "mayo_L506", "L506")) if code == 0 else []
    lits_preds = os.listdir(os.path.join(pred_root, "lits", "volume-0")) if code == 0 else []
    check("predicted 6 L506 slices and 2 LiTS slices", len(mayo_preds) == 6 and len(lits_preds) == 2)
    if mayo_preds:
        p = np.load(os.path.join(pred_root, "mayo_L506", "L506", sorted(mayo_preds)[0]))
        check("predictions are float32 HU, full slice size", p.dtype == np.float32 and p.shape == (128, 128))
    with open(os.path.join(pred_root, "mayo_L506", "timing.json")) as f:
        timing = json.load(f)
    print("timing.json:", timing)
    check("timing.json complete", set(timing) >= {"slices", "seconds_per_slice", "device", "epoch"}
          and timing["slices"] == 6 and timing["epoch"] == best["epoch"])

    # ---- evaluation ----
    code1, out1 = script("evaluate.py", "--model", "redcnn", *C)
    code2, out2 = script("evaluate.py", "--baseline", *C)
    for line in (out1 + out2).splitlines():
        if "PSNR" in line:
            print(" ", line)
    check("evaluate --model and --baseline finished", code1 == 0 and code2 == 0)
    check("results json files written",
          all(os.path.exists(os.path.join(runs, tag, f"results_{s}.json"))
              for tag in ("redcnn", "quarter_dose") for s in ("mayo_L506", "lits")))
    summary_path = os.path.join(results, "summary.csv")
    rows = read_csv(summary_path)
    check("summary.csv has 4 rows (2 models x 2 test sets)",
          sorted((r["model"], r["test_set"]) for r in rows) ==
          [("quarter_dose", "lits"), ("quarter_dose", "mayo_L506"), ("redcnn", "lits"), ("redcnn", "mayo_L506")])
    check("summary.csv columns", list(rows[0]) ==
          ["model", "test_set", "psnr", "ssim", "rmse", "seconds_per_slice", "device", "slices"])
    script("evaluate.py", "--baseline", *C)
    check("re-running evaluate replaces rows instead of duplicating", len(read_csv(summary_path)) == 4)

    # ---- run_all.sh: training is already done, so it goes straight to predict/evaluate ----
    code, out = run(["env", f"PYTHON={sys.executable}", "bash", "scripts/run_all.sh", "redcnn", *C])
    check("run_all.sh runs train --resume, predict, evaluate", code == 0 and "Already trained" in out
          and "Done." in out)

    # ---- a test set that is not prepared yet (e.g. LiTS) is skipped, not fatal ----
    no_lits = dict(cfg, data=dict(cfg["data"], test_sets=dict(
        cfg["data"]["test_sets"], lits={"root": os.path.join(tmp, "missing_lits"), "patients": "all"})))
    no_lits_path = os.path.join(tmp, "no_lits.yaml")
    with open(no_lits_path, "w") as f:
        yaml.safe_dump(no_lits, f)
    code, out = script("predict.py", "--model", "redcnn", "--config", no_lits_path)
    check("missing LiTS folder -> skipped with a warning", code == 0 and "skipping test set lits" in out)

    # ---- safety checks ----
    for split in ("train_patients", "val_patients"):
        bad = dict(cfg, data=dict(cfg["data"], **{split: ["L333", "L506"]}))
        bad_path = os.path.join(tmp, f"bad_{split}.yaml")
        with open(bad_path, "w") as f:
            yaml.safe_dump(bad, f)
        code, out = script("train.py", "--model", "redcnn", "--config", bad_path, expect_ok=False)
        check(f"L506 in {split} -> refuses to train", code != 0 and "Refusing to train" in out and "L506" in out)

    code, out = script("train.py", "--model", "no_such_model", *C, expect_ok=False)
    check("missing model -> message about template.py", code != 0 and "template.py" in out)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

check("temporary folder cleaned up", not os.path.exists(tmp))
check("real data/, runs/, results/ untouched",
      all(snapshot(d) == before for d, before in real_before.items()))

# ---- the real data, if it has been prepared on this computer ----
real = os.path.join(REPO, "data", "mayo", "L067")
if os.path.isdir(real):
    pairs = list_pairs(os.path.join(REPO, "data", "mayo"), ["L067"])
    lo, hi, patient, name = SlicePairs(pairs)[0]
    lo_hu, hi_hu = unit_to_hu(lo.numpy()), unit_to_hu(hi.numpy())
    print(f"real {patient}: {len(pairs)} slice pairs; {name}: shape {tuple(lo.shape)}, "
          f"quarter {lo_hu.min():.0f}..{lo_hu.max():.0f} HU, full {hi_hu.min():.0f}..{hi_hu.max():.0f} HU")
    check("real L067 slice is (1,512,512) in a CT-like HU range",
          tuple(lo.shape) == (1, 512, 512) and lo_hu.min() >= -1100 and lo_hu.max() <= 4000)
else:
    print("real data: data/mayo/L067 not found, skipped")

sys.exit(1 if failed else 0)
