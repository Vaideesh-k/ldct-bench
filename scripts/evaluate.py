"""Score predictions against the full-dose slices (PSNR, SSIM, RMSE).

    python scripts/evaluate.py --model redcnn          score runs/redcnn/predictions
    python scripts/evaluate.py --baseline              score the raw quarter-dose input
    python scripts/evaluate.py --model redcnn --sets mayo_L506

Writes runs/<model or quarter_dose>/results_<set>.json (means, stds, ...)
and updates results/summary.csv (one row per model and test set; committed to git).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# let this script import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import load_config, resolve
from ldct.data import choose_test_sets, test_set_pairs
from ldct.metrics import score

BASELINE = "quarter_dose"
SUMMARY_COLUMNS = ["model", "test_set", "psnr", "ssim", "rmse", "seconds_per_slice", "device", "slices"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="name of a trained model, e.g. redcnn")
    who.add_argument("--baseline", action="store_true", help="score the noisy quarter-dose input itself")
    ap.add_argument("--sets", nargs="+", help="test sets from the config (default: all whose folder exists)")
    ap.add_argument("--config", help="extra yaml merged on top of the configs (used by tests)")
    return ap.parse_args()


def score_set(cfg, set_name, pred_dir):
    """Score every slice of one test set. pred_dir=None scores the quarter-dose input."""
    lo, hi = cfg["eval"]["window"]
    try:
        pairs = test_set_pairs(cfg["data"]["test_sets"][set_name])
    except FileNotFoundError as e:
        raise SystemExit(f"Test set {set_name}: {e}")

    scores = []
    for lo_path, hi_path, patient, name in tqdm(pairs, desc=set_name):
        if pred_dir is None:
            pred = np.load(lo_path)
        else:
            path = os.path.join(pred_dir, patient, f"{name}.npy")
            if not os.path.exists(path):
                raise SystemExit(f"Missing prediction {path}. Run scripts/predict.py first.")
            pred = np.load(path)
        scores.append(score(pred, np.load(hi_path), lo, hi))
    return scores


def update_summary(path, new_row):
    """Add one row to summary.csv, replacing an older row for the same model and test set."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["model"], r["test_set"]) != (new_row["model"], new_row["test_set"])]
    rows.append(new_row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    tag = BASELINE if args.baseline else args.model
    cfg = load_config(tag, args.config)
    sets = choose_test_sets(cfg, args.sets)

    run_dir = os.path.join(resolve(cfg["output_dir"]), tag)
    summary_path = os.path.join(resolve(cfg["results_dir"]), "summary.csv")
    os.makedirs(run_dir, exist_ok=True)

    for set_name in sets:
        pred_dir = None if args.baseline else os.path.join(run_dir, "predictions", set_name)
        scores = score_set(cfg, set_name, pred_dir)

        timing = None
        if pred_dir and os.path.exists(os.path.join(pred_dir, "timing.json")):
            with open(os.path.join(pred_dir, "timing.json")) as f:
                timing = json.load(f)

        keys = ["psnr", "ssim", "rmse"]
        means = {k: float(np.mean([s[k] for s in scores])) for k in keys}
        stds = {k: float(np.std([s[k] for s in scores])) for k in keys}
        result = {"model": tag, "test_set": set_name, "slices": len(scores),
                  "window_hu": cfg["eval"]["window"], "mean": means, "std": stds, "timing": timing}
        with open(os.path.join(run_dir, f"results_{set_name}.json"), "w") as f:
            json.dump(result, f, indent=2)

        update_summary(summary_path, {
            "model": tag, "test_set": set_name,
            "psnr": f"{means['psnr']:.4f}", "ssim": f"{means['ssim']:.4f}", "rmse": f"{means['rmse']:.3f}",
            "seconds_per_slice": f"{timing['seconds_per_slice']:.4f}" if timing else "",
            "device": timing["device"] if timing else "",
            "slices": len(scores),
        })
        print(f"{tag} on {set_name} ({len(scores)} slices): "
              f"PSNR {means['psnr']:.2f}±{stds['psnr']:.2f} dB | "
              f"SSIM {means['ssim']:.4f}±{stds['ssim']:.4f} | "
              f"RMSE {means['rmse']:.2f}±{stds['rmse']:.2f} HU")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
