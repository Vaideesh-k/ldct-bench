"""Step 7 report figures.

    python scripts/make_figures.py --part local --out docs/report/figures     (laptop: committed results only)
    python scripts/make_figures.py --part zoom  --pred-root $SCR/pred ...     (DGX job slurm/figures.sbatch)
    python scripts/make_figures.py --part pick                                (print the chosen tumour's patient)

local:
  detection.png      20 LiTS test patients / 106 tumours: detected, detected < 10 mm, per-tumour Dice (95% CI)
  psnr_vs_dice.png   Mayo L506 PSNR against per-tumour Dice: the best image score is not the best Dice
  training_curves.png  training loss of control, 2D refiner, 2.5D refiner (same loss, same data)
zoom:
  small_tumour.png   one small tumour of the LiTS locked test set, zoomed (window -160..240 HU), every input,
                     with the ground-truth outline. Chosen by a rule fixed in advance, NOT by any model's
                     result: the largest tumour under 10 mm (equivalent diameter), its largest slice.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# palette (dataviz reference, light mode): reference inputs grey, published baselines blue, our models orange
GREY, BLUE, ORANGE, AQUA = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
GROUP = {"normal dose": GREY, "quarter dose": GREY, "CT-Mamba": BLUE, "CTformer": BLUE, "CoreDiff": BLUE,
         "control": ORANGE, "refiner": ORANGE, "2.5D refiner": ORANGE}
LABEL = {"refiner": "2D refiner", "control": "control (fine-tuned)"}

PANELS = [("normal dose", "full"), ("quarter dose", "quarter"), ("CT-Mamba", "ct_mamba"),
          ("CTformer", "ctformer"), ("CoreDiff", "corediff"), ("control", "ct_mamba_control"),
          ("2D refiner", "ct_mamba_refiner"), ("2.5D refiner", "ct_mamba_refiner25d")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- local figures
def detection_figure(res, out):
    rows = read_csv(os.path.join(res, "final_extended20_detection.csv"))
    extra = [r for r in read_csv(os.path.join(res, "final_extended20_25d_detection.csv")) if r["input"] == "2.5D refiner"]
    order = ["normal dose", "quarter dose", "CT-Mamba", "CTformer", "CoreDiff", "control", "refiner", "2.5D refiner"]
    by = {r["input"]: r for r in rows + extra}
    rows = [by[k] for k in order]
    y = np.arange(len(rows))[::-1]
    names = [LABEL.get(r["input"], r["input"]) for r in rows]
    colors = [GROUP[r["input"]] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    specs = [("tumours detected (of 106)", lambda r: int(r["detected"]), None, 106),
             ("tumours < 10 mm detected (of 57)", lambda r: int(r["small_detected"]), None, 57),
             ("per-tumour Dice (95% bootstrap CI)", lambda r: float(r["per_tumour_dice"]),
              lambda r: (float(r["dice_ci_low"]), float(r["dice_ci_high"])), None)]
    for ax, (title, val, ci, total) in zip(axes, specs):
        v = [val(r) for r in rows]
        ax.barh(y, v, height=0.62, color=colors, edgecolor="white", linewidth=2)
        if ci:
            lo, hi = zip(*[ci(r) for r in rows])
            ax.errorbar(v, y, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=INK2, lw=1, capsize=2)
        ref = v[0]
        ax.axvline(ref, color=INK2, lw=0.8, ls="--")
        for yy, vv in zip(y, v):
            ax.text((hi[list(y).index(yy)] if ci else vv) + (0.008 if ci else 0.8), yy,
                    f"{vv:.3f}" if ci else str(vv), va="center", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, (0.62 if ci else total * 0.8))
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, BLUE, ORANGE)]
    fig.legend(handles, ["reference (no denoiser)", "published denoisers", "this project (CT-Mamba fine-tuned)"],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("nnU-Net tumour detection on 20 locked LiTS test patients (dashed line = normal dose)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(out, "detection.png"), dpi=200)
    plt.close(fig)


def psnr_dice_figure(res, out):
    img = {r["input"]: r for r in read_csv(os.path.join(res, "final_extended20_image.csv"))}
    det = {r["input"]: r for r in read_csv(os.path.join(res, "final_extended20_detection.csv"))}
    det.update({r["input"]: r for r in read_csv(os.path.join(res, "final_extended20_25d_detection.csv"))})
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.axhline(float(det["normal dose"]["per_tumour_dice"]), color=INK2, lw=0.8, ls="--")
    ax.text(24.55, float(det["normal dose"]["per_tumour_dice"]) + 0.002, "normal dose", fontsize=7.5, color=INK2)
    offsets = {"quarter dose": (6, -3), "CT-Mamba": (7, -3), "CTformer": (-8, -13), "CoreDiff": (7, -3),
               "control": (-40, 7), "refiner": (7, -3), "2.5D refiner": (7, -3)}
    for name, (dx, dy) in offsets.items():
        x, yv = float(img[name]["psnr"]), float(det[name]["per_tumour_dice"])
        ax.scatter([x], [yv], s=46, color=GROUP[name], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(LABEL.get(name, name), (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("PSNR on Mayo L506, dB (image quality)", fontsize=8.5, color=INK)
    ax.set_ylabel("per-tumour Dice, 20 LiTS patients", fontsize=8.5, color=INK)
    ax.set_title("Higher PSNR does not mean better tumour outlines", fontsize=10, color=INK, loc="left")
    ax.set_xlim(24.2, 30.6)
    ax.set_ylim(0.348, 0.424)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "psnr_vs_dice.png"), dpi=200)
    plt.close(fig)


def training_figure(res, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    runs = [("ct_mamba_control", "control (CT-Mamba fine-tuned)", ORANGE),
            ("ct_mamba_refiner", "2D refiner", BLUE), ("ct_mamba_refiner25d", "2.5D refiner", AQUA)]
    for run, label, color in runs:
        f = os.path.join(res, f"{run}_train_log.csv")
        if not os.path.exists(f):
            continue
        rows = read_csv(f)
        it = [int(r["iteration"]) / 1000 for r in rows]
        loss = [float(r["window_loss"]) * 1000 for r in rows]
        ax.plot(it, loss, color=color, lw=2, label=label)
        ax.annotate(label, (it[-1], loss[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    ax.set_xlabel("iteration (thousands)", fontsize=8.5, color=INK)
    ax.set_ylabel("training loss x 1000 (L1 + tumour L1)", fontsize=8.5, color=INK)
    ax.set_title("Training loss, 1000-iteration means (same data, same loss)", fontsize=10, color=INK, loc="left")
    style(ax)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    ax.set_xticks(range(0, 31, 5))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- zoomed tumour (needs data + predictions)
def pick_tumour(lits, patients, small_mm=10.0):
    from scipy import ndimage
    from detect_eval import STRUCT, equivalent_diameter_mm, patient_spacing
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


def zoom_figure(args):
    test = [f"lits_{k:03d}" for k in json.load(open(args.split))["test"]]
    d, pid, name, cy, cx, outline = pick_tumour(args.lits, test)
    h = args.half
    r0, c0 = max(cy - h, 0), max(cx - h, 0)
    sl = (slice(r0, r0 + 2 * h), slice(c0, c0 + 2 * h))
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.9))
    for ax, (title, src) in zip(axes.ravel(), PANELS):
        if src in ("full", "quarter"):
            path = os.path.join(args.lits, pid, src, f"{name}.npy")
        else:
            root = os.path.join(args.pred_root, src) if args.pred_root else None
            cands = [os.path.join(root, "lits", pid, f"{name}.npy")] if root else []
            cands.append(os.path.join("runs", src, "predictions", "lits", pid, f"{name}.npy"))
            path = next((c for c in cands if os.path.exists(c)), cands[0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if os.path.exists(path):
            ax.imshow(np.load(path)[sl], cmap="gray", vmin=-160, vmax=240)
            ax.contour(outline[sl], levels=[0.5], colors="#eda100", linewidths=0.9)
        else:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            print("missing:", path)
    fig.suptitle(f"{pid}, slice {name}: tumour {d:.1f} mm (largest < 10 mm in the locked test set); "
                 f"yellow = ground-truth outline; window -160..240 HU", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "small_tumour.png"), dpi=200)
    print(f"small_tumour.png: {pid} {name}, {d:.1f} mm, centre ({cy}, {cx})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["local", "zoom", "pick"], default="local")
    ap.add_argument("--results", default="dgx_results", help="committed results (local part)")
    ap.add_argument("--lits", default="data/lits")
    ap.add_argument("--split", default="configs/lits_split.json")
    ap.add_argument("--pred-root", help="predictions at <pred-root>/<run>/lits/<patient> (else runs/<run>/predictions)")
    ap.add_argument("--out", default="docs/report/figures")
    ap.add_argument("--half", type=int, default=40, help="half width of the zoom in pixels")
    args = ap.parse_args()
    if args.part == "pick":
        test = [f"lits_{k:03d}" for k in json.load(open(args.split))["test"]]
        print(pick_tumour(args.lits, test)[1])
        return
    os.makedirs(args.out, exist_ok=True)
    if args.part == "zoom":
        zoom_figure(args)
        return
    detection_figure(args.results, args.out)
    psnr_dice_figure(args.results, args.out)
    training_figure(args.results, args.out)
    print("detection.png, psnr_vs_dice.png, training_curves.png written to", args.out)


if __name__ == "__main__":
    main()
