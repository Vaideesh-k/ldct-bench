"""Paired, per-lesion comparison of the detection results written by slurm/detect_all.sbatch.

Every input (normal dose, quarter dose, each model's output) is scored on the same ground-truth
lesions of the locked LiTS test patients, so the per-lesion Dice values are paired. This script
compares each input with normal dose and with quarter dose (Wilcoxon signed-rank test), lists
the lesions that are missed, and splits Dice by lesion size.

    python scripts/detect_compare.py --results dgx_results --out dgx_results/detect_paired.json
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

INPUTS = ["full", "quarter", "ct_mamba", "corediff", "ctformer"]


def lesions(result):
    pp = result["per_patient"]
    return [(p, i, l) for p in sorted(pp) for i, l in enumerate(pp[p]["per_lesion"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="folder with detect_<input>.json")
    ap.add_argument("--inputs", nargs="+", default=INPUTS)
    ap.add_argument("--small-mm", type=float, default=10.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    res = {n: json.load(open(os.path.join(args.results, f"detect_{n}.json"))) for n in args.inputs}
    ref = lesions(res["full"])
    for n in args.inputs:   # same lesions in the same order for every input
        assert [(p, i, l["voxels"]) for p, i, l in lesions(res[n])] == [(p, i, l["voxels"]) for p, i, l in ref], n
    dice = {n: np.array([l["dice"] for _, _, l in lesions(res[n])]) for n in args.inputs}
    det = {n: np.array([l["detected"] for _, _, l in lesions(res[n])]) for n in args.inputs}
    diam = np.array([l["diameter_mm"] for _, _, l in ref])
    small = diam < args.small_mm

    out = {"lesions": len(ref), "small_lesions": int(small.sum()), "small_mm": args.small_mm,
           "per_input": {}, "paired": {}, "missed": []}
    for n in args.inputs:
        out["per_input"][n] = {"mean_lesion_dice": float(dice[n].mean()),
                               "mean_dice_small": float(dice[n][small].mean()),
                               "mean_dice_large": float(dice[n][~small].mean()),
                               "detected": int(det[n].sum())}
    for base in ("full", "quarter"):
        for n in args.inputs:
            if n == base:
                continue
            d = dice[n] - dice[base]
            out["paired"][f"{n}_vs_{base}"] = {
                "mean_dice_difference": float(d.mean()),
                "lesions_better": int((d > 0.01).sum()), "lesions_worse": int((d < -0.01).sum()),
                "wilcoxon_p": float(wilcoxon(dice[n], dice[base]).pvalue)}
    for j, (p, i, l) in enumerate(ref):
        if not all(det[n][j] for n in args.inputs):
            out["missed"].append({"patient": p, "lesion": i, "diameter_mm": round(l["diameter_mm"], 1),
                                  "detected_by": [n for n in args.inputs if det[n][j]]})

    for n, v in out["per_input"].items():
        print(f"{n:9s} detected {v['detected']}/{len(ref)}  lesion Dice {v['mean_lesion_dice']:.3f}  "
              f"small {v['mean_dice_small']:.3f}  large {v['mean_dice_large']:.3f}")
    for k, v in out["paired"].items():
        print(f"{k:22s} {v['mean_dice_difference']:+.3f}  better/worse {v['lesions_better']}/{v['lesions_worse']}  "
              f"p={v['wilcoxon_p']:.3f}")
    for m in out["missed"]:
        print(f"missed {m['patient']} #{m['lesion']} {m['diameter_mm']} mm, found by {m['detected_by'] or 'none'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
