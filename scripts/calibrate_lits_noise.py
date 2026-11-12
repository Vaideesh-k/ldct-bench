"""Measure the noise the real Mayo quarter-dose scans add, so the LiTS simulation
can be matched to it.

    python scripts/calibrate_lits_noise.py --mayo data/mayo [--every 10]
    python scripts/calibrate_lits_noise.py --lits data/lits/lits_000 [more patient dirs]

--mayo: for the TRAINING patients (never L506), take every k-th slice, find uniform
soft-tissue regions in the full-dose slice (local std of a smoothed copy < 4 HU, mean
30..130 HU = liver-like) and report the std of (quarter - full) there. Because the
Mayo quarter dose is the full dose plus inserted projection noise, quarter - full IS
the inserted noise, directly comparable with our simulation's (low - full).
--lits: the same statistic from a converted patient, in the liver mask.
Prints median / quartiles; the median over slices is the calibration target.
Also prints the std of the full-dose image itself in those regions (for context).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from ldct.config import load_config, resolve                              # noqa: E402
from make_lits_lowdose import added_noise_std, liver_roi                  # noqa: E402


def uniform_soft_tissue(full, lo=30, hi=130, max_std=4.0, smooth=2.0, win=9):
    """Boolean ROI of flat, liver-like pixels in a full-dose slice."""
    sm = gaussian_filter(full.astype(np.float64), smooth)
    mean = uniform_filter(sm, win)
    sq = uniform_filter(sm ** 2, win)
    local_std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    roi = (local_std < max_std) & (mean > lo) & (mean < hi)
    roi[:win, :] = roi[-win:, :] = roi[:, :win] = roi[:, -win:] = False
    return roi


def summarize(label, values, extra=""):
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        print(f"{label}: no usable slices")
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    print(f"{label}: {len(v)} slices | added-noise std median {med:.2f} HU (IQR {q1:.2f}-{q3:.2f}) {extra}")
    return float(med)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mayo", help="Mayo root, e.g. data/mayo")
    ap.add_argument("--lits", nargs="+", help="converted LiTS patient folders")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--json", help="write the numbers here too")
    args = ap.parse_args()
    out = {}

    if args.mayo:
        cfg = load_config("quarter_dose")
        stds, full_stds, per_patient = [], [], {}
        for p in cfg["data"]["train_patients"]:
            files = sorted(glob.glob(os.path.join(str(resolve(args.mayo)), p, "full", "*.npy")))[::args.every]
            ps = []
            for f in files:
                full = np.load(f).astype(np.float64)
                quarter = np.load(f.replace("/full/", "/quarter/")).astype(np.float64)
                roi = uniform_soft_tissue(full)
                if roi.sum() < 2000:
                    continue
                ps.append(added_noise_std(quarter, full, roi))
                full_stds.append(float(full[roi].std()))
            per_patient[p] = summarize(f"  Mayo {p}", ps)
            stds += ps
        med = summarize("Mayo train patients, uniform soft tissue", stds,
                        f"| full-dose std there {np.median(full_stds):.2f} HU")
        out["mayo"] = {"added_noise_std_median_hu": med, "per_patient": per_patient,
                       "full_dose_std_median_hu": float(np.median(full_stds)), "slices": len(stds)}

    if args.lits:
        for d in args.lits:
            stds, full_stds = [], []
            for f in sorted(glob.glob(os.path.join(d, "full", "*.npy"))):
                full = np.load(f).astype(np.float64)
                low = np.load(f.replace("/full/", "/quarter/")).astype(np.float64)
                roi = liver_roi(np.load(f.replace("/full/", "/mask/")))
                if roi.sum() < 2000:
                    continue
                stds.append(added_noise_std(low, full, roi))
                full_stds.append(float(full[roi].std()))
            meta = json.load(open(os.path.join(d, "meta.json")))
            med = summarize(f"LiTS {os.path.basename(d)} (I0 {meta['simulation']['I0']:.3g}), liver", stds,
                            f"| original std there {np.median(full_stds) if full_stds else float('nan'):.2f} HU")
            out[os.path.basename(d)] = {"I0": meta["simulation"]["I0"], "added_noise_std_median_hu": med, "slices": len(stds)}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
