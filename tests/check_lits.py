"""Checks for scripts/make_lits_lowdose.py on synthetic data (no LiTS download needed)."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from make_lits_lowdose import (added_noise_std, axial, convert_patient, liver_roi, liver_slice_range,
                               simulate_low_dose)

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


# ---- a water phantom with a "liver" disc, 128 px, 1 mm pixels ----
n = 128
yy, xx = np.mgrid[:n, :n]
body = (xx - 64) ** 2 + (yy - 64) ** 2 < 55 ** 2
liver = (xx - 48) ** 2 + (yy - 60) ** 2 < 18 ** 2
hu = np.full((n, n), -1024.0)
hu[body] = 0.0
hu[liver] = 60.0

low = simulate_low_dose(hu, 1.0, I0=2e5, dose=0.25, n_angles=360, seed=1)
low2 = simulate_low_dose(hu, 1.0, I0=2e5, dose=0.25, n_angles=360, seed=1)
low3 = simulate_low_dose(hu, 1.0, I0=2e5, dose=0.25, n_angles=360, seed=2)
check("same shape and float32", low.shape == hu.shape and low.dtype == np.float32)
check("same seed -> same image; other seed -> different", np.array_equal(low, low2) and not np.array_equal(low, low3))
roi = liver_roi(np.where(liver, 1, 0).astype(np.uint8), erode=3)
bias = float((low - hu)[roi].mean())
std = added_noise_std(low, hu, roi)
check(f"added noise is zero-mean in the liver (bias {bias:.2f} HU)", abs(bias) < 3)
check(f"added noise has a sensible size (std {std:.1f} HU)", 3 < std < 200)
std_hi = added_noise_std(simulate_low_dose(hu, 1.0, I0=8e5, dose=0.25, n_angles=360, seed=1), hu, roi)
check(f"4x the photons -> about half the noise ({std:.1f} -> {std_hi:.1f} HU)", 0.4 < std_hi / std < 0.6)
std_half = added_noise_std(simulate_low_dose(hu, 1.0, I0=2e5, dose=0.5, n_angles=360, seed=1), hu, roi)
check(f"half dose adds less noise than quarter dose ({std_half:.1f} < {std:.1f} HU)", std_half < std)
check("air outside the body stays air-like", abs(float(low[~body].mean()) + 1024) < 15)
check("small ROI -> nan", np.isnan(added_noise_std(low, hu, np.zeros_like(roi))))

# ---- orientation + slice range on a synthetic volume ----
vol = np.zeros((6, 8, 10), np.float32)          # (x, y, z)
vol[5, 0, 3] = 1.0                              # max x (patient's right), min y (posterior)
sl = axial(vol, 3)
check("axial(): shape (y, x) = (8, 6)", sl.shape == (8, 6))
check("axial(): patient's right & posterior -> bottom-left of the image", sl[-1, 0] == 1.0 and sl.sum() == 1.0)
mask = np.zeros((6, 8, 10), np.int16)
mask[2:4, 2:4, 4:7] = 1
check("liver_slice_range with margin 2 -> (2, 8)", liver_slice_range(mask, 2) == (2, 8))
check("liver_slice_range clamps at the volume ends", liver_slice_range(mask, 10) == (0, 9))

# ---- whole conversion of a tiny fake patient ----
try:
    import nibabel as nib
    tmp = tempfile.mkdtemp()
    affine = np.diag([0.8, 0.8, 2.5, 1.0])
    v = np.full((64, 64, 12), -1024.0, np.float32)
    v[16:48, 16:48, :] = 0.0
    m = np.zeros((64, 64, 12), np.uint8)
    m[24:40, 24:40, 4:8] = 1
    m[30:34, 30:34, 5:7] = 2
    v[24:40, 24:40, 4:8] = 60.0
    nib.save(nib.Nifti1Image(v, affine), os.path.join(tmp, "volume-7.nii"))
    nib.save(nib.Nifti1Image(m, affine), os.path.join(tmp, "segmentation-7.nii"))
    out = os.path.join(tmp, "lits")
    meta = convert_patient(os.path.join(tmp, "volume-7.nii"), os.path.join(tmp, "segmentation-7.nii"),
                           out, "lits_007", I0=2e5, margin=2, workers=1, seed=7)
    kept = sorted(os.listdir(os.path.join(out, "lits_007", "full")))
    check("kept slices 2..9 (liver 4-7 plus margin 2)", meta["kept_slices"] == {"first": 2, "last": 9, "count": 8} and len(kept) == 8)
    f0 = np.load(os.path.join(out, "lits_007", "full", kept[0]))
    q0 = np.load(os.path.join(out, "lits_007", "quarter", kept[0]))
    m2 = np.load(os.path.join(out, "lits_007", "mask", kept[3]))   # original z = 5: liver + tumour
    check("full/quarter are int16 HU, mask uint8", f0.dtype == np.int16 and q0.dtype == np.int16 and m2.dtype == np.uint8)
    check("mask keeps labels 1 and 2", set(np.unique(m2)) == {0, 1, 2})
    check("quarter differs from full where there is tissue", (q0 != f0).any())
    with open(os.path.join(out, "lits_007", "meta.json")) as fh:
        saved = json.load(fh)
    check("meta.json: spacing and I0 recorded", saved["spacing_mm"]["z"] == 2.5 and saved["simulation"]["I0"] == 2e5)
    check("meta.json: tumour voxels counted", saved["tumour_voxels"] == 4 * 4 * 2)
except ImportError:
    print("nibabel missing: skipped the NIfTI conversion checks")

print("\nFAILED:" if failed else "\nAll LiTS checks passed.", failed or "")
sys.exit(1 if failed else 0)
