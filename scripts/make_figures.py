"""Step 7 figures (run on the DGX, where the slices and predictions are):

    python scripts/make_figures.py --out results/figures

1. small_tumour.png: one small tumour of the LiTS locked test set, zoomed, as normal dose,
   quarter dose, CT-Mamba, CTformer, CoreDiff, control, refiner (window -160..240 HU), with
   the ground-truth outline. The tumour is chosen by a rule fixed in advance, NOT by any
   model's result: the largest tumour under 10 mm (equivalent diameter), its largest slice.
2. training_curves.png: window loss of the refiner and the control (same loss, same data).
Inputs that do not exist yet are drawn as empty panels.
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_eval import STRUCT, equivalent_diameter_mm, patient_spacing   # noqa: E402

PANELS = [("normal dose", "full"), ("quarter dose", "quarter"), ("CT-Mamba", "runs/ct_mamba"),
          ("CTformer", "runs/ctformer"), ("CoreDiff", "runs/corediff"),
          ("control", "runs/ct_mamba_control"), ("refiner", "runs/ct_mamba_refiner")]


def pick_tumour(lits, patients, small_mm=10.0):
    best = None
    for pid in patients:
        files = sorted(glob.glob(os.path.join(lits, pid, "mask", "*.npy")))
        vol = np.stack([np.load(f) for f in files]) == 2
        lab, n = ndimage.label(vol, STRUCT)
        sp = patient_spacing(os.path.join(lits, pid))
        for k in range(1, n + 1):
            comp = lab == k
            d = equivalent_diameter_mm(comp.sum(), sp)
            if d < small_mm and (best is None or d > best[0]):
                z = int(np.argmax(comp.sum(axis=(1, 2))))
                cy, cx = (int(round(c)) for c in ndimage.center_of_mass(comp[z]))
                best = (d, pid, os.path.basename(files[z])[:-4], cy, cx, comp[z])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lits", default="data/lits")
    ap.add_argument("--split", default="configs/lits_split.json")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--half", type=int, default=40, help="half width of the zoom in pixels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    test = [f"lits_{k:03d}" for k in json.load(open(args.split))["test"]]
    d, pid, name, cy, cx, outline = pick_tumour(args.lits, test)
    h = args.half
    r0, c0 = max(cy - h, 0), max(cx - h, 0)
    sl = (slice(r0, r0 + 2 * h), slice(c0, c0 + 2 * h))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(2.2 * len(PANELS), 2.6))
    for ax, (title, src) in zip(axes, PANELS):
        path = (os.path.join(args.lits, pid, src, f"{name}.npy") if not src.startswith("runs/")
                else os.path.join(src, "predictions", "lits", pid, f"{name}.npy"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if os.path.exists(path):
            ax.imshow(np.load(path)[sl], cmap="gray", vmin=-160, vmax=240)
            ax.contour(outline[sl], levels=[0.5], colors="#e8a33d", linewidths=0.8)
        else:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=8, transform=ax.transAxes)
    fig.suptitle(f"{pid} slice {name}: tumour {d:.1f} mm (largest < 10 mm in the test set; outline = ground truth)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "small_tumour.png"), dpi=200)
    print(f"small_tumour.png: {pid} {name}, {d:.1f} mm, centre ({cy}, {cx})")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for run, label, color in (("ct_mamba_refiner", "refiner", "#2f6db5"), ("ct_mamba_control", "control", "#c2541b")):
        f = os.path.join("runs", run, "train_log.csv")
        if os.path.exists(f):
            rows = list(csv.DictReader(open(f)))
            ax.plot([int(r["iteration"]) for r in rows], [float(r["window_loss"]) for r in rows], color=color, lw=2, label=label)
    ax.set_xlabel("iteration (262,144 training pixels each)")
    ax.set_ylabel("training loss (L1 + lesion L1), 1000-iteration mean")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_curves.png"), dpi=200)
    print("training_curves.png written")


if __name__ == "__main__":
    main()
