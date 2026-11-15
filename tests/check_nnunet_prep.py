"""Checks for scripts/prep_nnunet.py on a fake data/lits (needs nibabel)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


tmp = tempfile.mkdtemp()
lits = os.path.join(tmp, "lits")
for pid in ("lits_000", "lits_002"):
    for kind in ("full", "quarter", "mask"):
        os.makedirs(os.path.join(lits, pid, kind))
        for i in range(3):
            arr = np.zeros((16, 16), np.uint8) if kind == "mask" else np.full((16, 16), 40, np.int16)
            np.save(os.path.join(lits, pid, kind, f"{pid}_{i:04d}.npy"), arr)
    json.dump({"spacing_mm": {"x": 0.8, "y": 0.8, "z": 2.0}}, open(os.path.join(lits, pid, "meta.json"), "w"))
split = os.path.join(tmp, "split.json")
json.dump({"train": [0], "test": [2]}, open(split, "w"))
script = os.path.join(ROOT, "scripts", "prep_nnunet.py")


def run(*args):
    return subprocess.run([sys.executable, script, "--lits", lits, "--split", split, *args], capture_output=True, text=True)


r = run("--out", os.path.join(tmp, "raw"))
ds = os.path.join(tmp, "raw", "Dataset501_LiTSLiver")
check("train dataset: image, label and dataset.json written", r.returncode == 0
      and os.path.exists(os.path.join(ds, "imagesTr", "lits_000_0000.nii.gz"))
      and os.path.exists(os.path.join(ds, "labelsTr", "lits_000.nii.gz")) and os.path.exists(os.path.join(ds, "dataset.json")))
check("train dataset: only train patients", not os.path.exists(os.path.join(ds, "imagesTr", "lits_002_0000.nii.gz")))
meta = json.load(open(os.path.join(ds, "dataset.json")))
check("dataset.json: labels 0/1/2, 1 training case", meta["labels"] == {"background": 0, "liver": 1, "tumour": 2} and meta["numTraining"] == 1)
import nibabel as nib
img = nib.load(os.path.join(ds, "imagesTr", "lits_000_0000.nii.gz"))
check("NIfTI is (x, y, z) with the meta spacing", img.shape == (16, 16, 3)
      and tuple(round(float(z), 2) for z in img.header.get_zooms()) == (0.8, 0.8, 2.0))

r = run("--part", "test", "--images-only", "--out", os.path.join(tmp, "tq"), "--from-predictions", lits, "--from-kind", "quarter")
check("test images only (quarter kind), test patients only", r.returncode == 0
      and os.path.exists(os.path.join(tmp, "tq", "lits_002_0000.nii.gz")) and not os.path.exists(os.path.join(tmp, "tq", "lits_000_0000.nii.gz")))
r = run("--part", "test", "--out", os.path.join(tmp, "bad"))
check("refuses to write labels for the locked test patients", r.returncode != 0 and "never written" in r.stderr)
r = run("--part", "test", "--images-only", "--out", os.path.join(tmp, "tp"), "--from-predictions", os.path.join(tmp, "nowhere"))
check("missing predictions: skipped, not an error", r.returncode == 0 and "skipped" in r.stdout)

print("\nFAILED:" if failed else "\nAll nnU-Net prep checks passed.", failed or "")
sys.exit(1 if failed else 0)
