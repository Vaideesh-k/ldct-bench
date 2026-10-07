"""Denoise every test slice with a trained model and time it.

    python scripts/predict.py --model redcnn                       all test sets in the config
    python scripts/predict.py --model redcnn --sets mayo_L506      only some test sets
    python scripts/predict.py --model redcnn --checkpoint latest.pth

Writes runs/<model>/predictions/<set>/<patient>/<name>.npy (float32, HU)
and runs/<model>/predictions/<set>/timing.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

# let this script import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import load_config, resolve
from ldct.data import choose_test_sets, hu_to_unit, test_set_pairs, unit_to_hu
from ldct.infer import denoise_slice
from ldct.models import build_model
from ldct.runtime import device_name, pick_device, synchronize


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="name of a file in ldct/models/, e.g. redcnn")
    ap.add_argument("--sets", nargs="+", help="test sets from the config (default: all whose folder exists)")
    ap.add_argument("--checkpoint", default="best.pth", help="file in runs/<model>/ (default best.pth)")
    ap.add_argument("--config", help="extra yaml merged on top of the configs (used by tests)")
    return ap.parse_args()


def load_trained_model(cfg, model_name, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"{ckpt_path} not found. Train first: python scripts/train.py --model {model_name}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, **ckpt.get("model_args", cfg["model_args"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("epoch")


def predict_set(model, cfg, set_name, out_dir, device):
    """Denoise all slices of one test set; returns the seconds each slice took."""
    try:
        pairs = test_set_pairs(cfg["data"]["test_sets"][set_name])
    except FileNotFoundError as e:
        raise SystemExit(f"Test set {set_name}: {e}")

    seconds = []
    for lo_path, _, patient, name in tqdm(pairs, desc=set_name):
        x = torch.from_numpy(hu_to_unit(np.load(lo_path).astype(np.float32)))[None, None].to(device)

        synchronize(device)               # make sure nothing else is still running
        start = time.perf_counter()
        out = denoise_slice(model, x, cfg)
        synchronize(device)               # wait until the GPU has really finished
        seconds.append(time.perf_counter() - start)

        pred_hu = unit_to_hu(out[0, 0].float().cpu().numpy()).astype(np.float32)
        os.makedirs(os.path.join(out_dir, patient), exist_ok=True)
        np.save(os.path.join(out_dir, patient, f"{name}.npy"), pred_hu)
    return seconds


def main():
    args = parse_args()
    cfg = load_config(args.model, args.config)
    sets = choose_test_sets(cfg, args.sets)
    run_dir = os.path.join(resolve(cfg["output_dir"]), args.model)

    device = pick_device(cfg["device"])
    model, epoch = load_trained_model(cfg, args.model, os.path.join(run_dir, args.checkpoint), device)
    print(f"Model {args.model}, {args.checkpoint} (epoch {epoch}), device {device_name(device)}")

    for set_name in sets:
        out_dir = os.path.join(run_dir, "predictions", set_name)
        seconds = predict_set(model, cfg, set_name, out_dir, device)

        # the first slice includes one-off GPU start-up costs, so it is left out
        timed = seconds[1:] if len(seconds) > 1 else seconds
        timing = {
            "slices": len(seconds),
            "seconds_per_slice": sum(timed) / len(timed),
            "device": device_name(device),
            "epoch": epoch,
            "checkpoint": args.checkpoint,
        }
        with open(os.path.join(out_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)
        print(f"{set_name}: {len(seconds)} slices -> {out_dir} "
              f"({timing['seconds_per_slice'] * 1000:.1f} ms per slice)")


if __name__ == "__main__":
    main()
