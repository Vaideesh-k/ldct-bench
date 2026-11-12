"""Lesion-wise detection metrics for liver tumours: how many of the real tumours does a
predicted segmentation find, especially the small ones?

    python scripts/detect_eval.py --pred <dir or file> --gt <dir or file> [--spacing dz dy dx]
    python scripts/detect_eval.py --lits data/lits --pred-root runs/<model>/nnunet_pred  --out results/detect_<model>.json

Inputs are 3D label volumes (tumour = label 2, or any value >= 1 with --binary):
either NIfTI/.npy volumes, or a folder of per-slice .npy files as data/lits/<id>/mask/.

Definitions (all 3D, 26-connectivity for the ground truth, same for the prediction):
  lesion            one connected component of the ground-truth tumour mask
  detected          a lesion with at least one voxel overlapped by the prediction
                    (any overlap counts; use --min-overlap to demand a fraction)
  detection rate    detected lesions / all lesions, overall and for lesions with an
                    equivalent diameter < 10 mm; the equivalent diameter is that of a
                    sphere with the lesion's volume (voxel count x voxel volume)
  false positive    a predicted component that overlaps no ground-truth lesion;
                    reported as the mean count per patient
  per-lesion Dice   Dice of the prediction restricted to the lesion's bounding box
                    (dilated by 1 voxel) with that lesion
  per-patient Dice  Dice of the whole tumour masks of one patient
Results are printed and written as JSON (per patient and per lesion lists included).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage

TUMOUR = 2
STRUCT = np.ones((3, 3, 3), bool)      # 26-connectivity


# ----------------------------------------------------------------------------- loading
def load_volume(path):
    """A 3D integer volume from a .nii/.nii.gz, a 3D .npy, or a folder of 2D .npy slices.
    Slice folders are stacked in sorted order along axis 0 (z, y, x)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npy")))
        if not files:
            raise FileNotFoundError(f"no .npy slices in {path}")
        return np.stack([np.load(f) for f in files])
    if path.endswith(".npy"):
        return np.load(path)
    import nibabel as nib
    return np.asanyarray(nib.load(path).dataobj)


def tumour_mask(vol, binary=False):
    return np.asarray(vol) >= 1 if binary else np.asarray(vol) == TUMOUR


# ----------------------------------------------------------------------------- metrics
def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def equivalent_diameter_mm(n_voxels, spacing):
    """Diameter of the sphere with the same volume as n_voxels voxels of size `spacing` (mm)."""
    volume = float(n_voxels) * float(np.prod(spacing))
    return float((6.0 * volume / np.pi) ** (1.0 / 3.0))


def evaluate_patient(pred, gt, spacing, min_overlap=0.0, small_mm=10.0):
    """Lesion-wise results for one patient. pred and gt are boolean 3D tumour masks."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction {pred.shape} and ground truth {gt.shape} differ in shape")
    gt_lab, n_gt = ndimage.label(gt, STRUCT)
    pr_lab, n_pr = ndimage.label(pred, STRUCT)

    lesions = []
    hit_pred_ids = set()
    objects = ndimage.find_objects(gt_lab)
    for i in range(1, n_gt + 1):
        sl = objects[i - 1]
        box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)   # dilated by 1 voxel
        lesion = gt_lab[box] == i
        n_vox = int(lesion.sum())
        overlap = pred[box] & lesion
        frac = overlap.sum() / n_vox
        detected = bool(frac > min_overlap) if min_overlap > 0 else bool(overlap.any())
        if overlap.any():
            hit_pred_ids.update(int(v) for v in np.unique(pr_lab[box][overlap]) if v)
        lesions.append({
            "voxels": n_vox,
            "diameter_mm": equivalent_diameter_mm(n_vox, spacing),
            "detected": detected,
            "overlap_fraction": float(frac),
            "dice": float(dice(pred[box], lesion)),
        })
    false_positives = int(n_pr - len(hit_pred_ids))
    small = [l for l in lesions if l["diameter_mm"] < small_mm]
    return {
        "lesions": len(lesions),
        "detected": int(sum(l["detected"] for l in lesions)),
        "small_lesions": len(small),
        "small_detected": int(sum(l["detected"] for l in small)),
        "false_positives": false_positives,
        "dice": float(dice(pred, gt)),
        "per_lesion": lesions,
    }


def summarize(per_patient, small_mm=10.0):
    """Pool the patients: detection rates over all lesions, FP per patient, mean Dices."""
    n = sum(p["lesions"] for p in per_patient.values())
    d = sum(p["detected"] for p in per_patient.values())
    ns = sum(p["small_lesions"] for p in per_patient.values())
    ds = sum(p["small_detected"] for p in per_patient.values())
    lesion_dices = [l["dice"] for p in per_patient.values() for l in p["per_lesion"]]
    return {
        "patients": len(per_patient),
        "lesions": n, "detected": d,
        "detection_rate": d / n if n else None,
        f"small_lesions_under_{small_mm:g}mm": ns, "small_detected": ds,
        "small_detection_rate": ds / ns if ns else None,
        "false_positives_per_patient": float(np.mean([p["false_positives"] for p in per_patient.values()])) if per_patient else None,
        "mean_per_lesion_dice": float(np.mean(lesion_dices)) if lesion_dices else None,
        "mean_per_patient_dice": float(np.mean([p["dice"] for p in per_patient.values()])) if per_patient else None,
    }


# ----------------------------------------------------------------------------- CLI
def patient_spacing(lits_patient_dir):
    """(dz, dy, dx) in mm from data/lits/<id>/meta.json (slices stack along z)."""
    with open(os.path.join(lits_patient_dir, "meta.json")) as f:
        s = json.load(f)["spacing_mm"]
    return (s["z"], s["y"], s["x"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="one prediction volume (file or slice folder)")
    ap.add_argument("--gt", help="one ground-truth volume (file or slice folder)")
    ap.add_argument("--spacing", nargs=3, type=float, metavar=("DZ", "DY", "DX"), help="voxel size in mm for --pred/--gt")
    ap.add_argument("--lits", help="data/lits root: evaluate every patient that has a prediction")
    ap.add_argument("--pred-root", help="folder with <patient id>.npy / .nii.gz / <id>/ slice folders")
    ap.add_argument("--patients", nargs="+", help="restrict to these patient ids (e.g. the locked test set)")
    ap.add_argument("--binary", action="store_true", help="treat any label >= 1 as tumour (default: label 2 only)")
    ap.add_argument("--min-overlap", type=float, default=0.0, help="fraction of a lesion that must be covered to count as detected")
    ap.add_argument("--small-mm", type=float, default=10.0)
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args()

    per_patient = {}
    if args.pred and args.gt:
        if not args.spacing:
            raise SystemExit("--spacing DZ DY DX is required with --pred/--gt")
        per_patient["volume"] = evaluate_patient(tumour_mask(load_volume(args.pred), args.binary),
                                                 tumour_mask(load_volume(args.gt), args.binary),
                                                 args.spacing, args.min_overlap, args.small_mm)
    elif args.lits and args.pred_root:
        ids = args.patients or sorted(d for d in os.listdir(args.lits) if os.path.isdir(os.path.join(args.lits, d)))
        for pid in ids:
            cands = [os.path.join(args.pred_root, pid + ext) for ext in (".npy", ".nii.gz", ".nii", "")]
            pred_path = next((c for c in cands if os.path.exists(c)), None)
            if pred_path is None:
                print(f"{pid}: no prediction, skipped")
                continue
            gt = tumour_mask(load_volume(os.path.join(args.lits, pid, "mask")), args.binary)
            pred = tumour_mask(load_volume(pred_path), args.binary)
            per_patient[pid] = evaluate_patient(pred, gt, patient_spacing(os.path.join(args.lits, pid)),
                                                args.min_overlap, args.small_mm)
            r = per_patient[pid]
            print(f"{pid}: {r['detected']}/{r['lesions']} lesions found ({r['small_detected']}/{r['small_lesions']} small), "
                  f"{r['false_positives']} FP, Dice {r['dice']:.3f}")
    else:
        raise SystemExit("give --pred and --gt, or --lits and --pred-root")

    summary = summarize(per_patient, args.small_mm)
    print(json.dumps(summary, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_patient": per_patient}, f, indent=1)


if __name__ == "__main__":
    main()
