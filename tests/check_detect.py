"""Checks for scripts/detect_eval.py on synthetic 3D masks."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from detect_eval import dice, equivalent_diameter_mm, evaluate_patient, load_volume, summarize, tumour_mask

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


def ball(shape, centre, r):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= r ** 2


shape = (40, 64, 64)
spacing = (2.0, 1.0, 1.0)             # dz, dy, dx in mm
big = ball(shape, (20, 20, 20), 6)    # ~ 12 vox across = 12 mm in-plane, 24 mm in z -> big
small = ball(shape, (20, 45, 45), 2)  # tiny
single = np.zeros(shape, bool); single[5, 5, 5] = True   # 1 voxel = 2 mm^3
gt = big | small | single

# equivalent diameter: 1 voxel of 2 mm^3 -> (6*2/pi)^(1/3) = 1.563 mm
check("equivalent diameter of one 2 mm^3 voxel", abs(equivalent_diameter_mm(1, spacing) - 1.5632) < 1e-3)
check("dice of identical masks = 1, of disjoint = 0, of empty = 1",
      dice(gt, gt) == 1.0 and dice(big, small) == 0.0 and dice(np.zeros(3, bool), np.zeros(3, bool)) == 1.0)

# perfect prediction
r = evaluate_patient(gt, gt, spacing)
check("3 lesions found (26-connectivity)", r["lesions"] == 3 and r["detected"] == 3)
check("2 of them are under 10 mm", r["small_lesions"] == 2 and r["small_detected"] == 2)
check("no false positives, Dice 1", r["false_positives"] == 0 and r["dice"] == 1.0)
check("per-lesion Dice 1", all(l["dice"] == 1.0 for l in r["per_lesion"]))

# prediction finds only the big lesion, plus one blob elsewhere (false positive)
pred = big.copy()
pred[30:33, 50:53, 10:13] = True
r = evaluate_patient(pred, gt, spacing)
check("only the big lesion detected", r["detected"] == 1 and r["small_detected"] == 0)
check("one false positive", r["false_positives"] == 1)
check("per-lesion Dice of the missed ones is 0", sorted(l["dice"] for l in r["per_lesion"])[:2] == [0.0, 0.0])

# a single predicted voxel touching the small lesion counts as detected, no FP
pred = np.zeros(shape, bool); pred[20, 45, 45] = True
r = evaluate_patient(pred, gt, spacing)
check("one-voxel overlap detects the small lesion", r["detected"] == 1 and r["small_detected"] == 1 and r["false_positives"] == 0)
check("...but not with --min-overlap 0.5", evaluate_patient(pred, gt, spacing, min_overlap=0.5)["detected"] == 0)

# one predicted blob covering two lesions: both detected, no FP
pred = ball(shape, (20, 32, 32), 20)
r = evaluate_patient(pred, gt, spacing)
check("one big blob covering 2 lesions: 2 detected, 0 FP", r["detected"] == 2 and r["false_positives"] == 0)

# pooled summary
s = summarize({"a": evaluate_patient(gt, gt, spacing), "b": evaluate_patient(np.zeros(shape, bool), gt, spacing)})
check("summary: 3/6 detected, 2/4 small, 0 FP per patient", s["detection_rate"] == 0.5 and s["small_detection_rate"] == 0.5
      and s["false_positives_per_patient"] == 0.0)
check("summary: per-patient Dice mean of 1 and 0", s["mean_per_patient_dice"] == 0.5)

# shape mismatch is an error; label handling; loading from a slice folder
try:
    evaluate_patient(gt[:-1], gt, spacing); check("shape mismatch raises", False)
except ValueError:
    check("shape mismatch raises", True)
lab = np.where(gt, 2, 0); lab[0, 0, 0] = 1   # liver voxel
check("tumour_mask: label 2 only, or >=1 with binary", tumour_mask(lab).sum() == gt.sum() and tumour_mask(lab, True).sum() == gt.sum() + 1)
tmp = tempfile.mkdtemp()
for z in range(shape[0]):
    np.save(os.path.join(tmp, f"p_{z:04d}.npy"), lab[z].astype(np.uint8))
check("load_volume stacks a slice folder along z", np.array_equal(load_volume(tmp), lab))

print("\nFAILED:" if failed else "\nAll detection checks passed.", failed or "")
sys.exit(1 if failed else 0)
