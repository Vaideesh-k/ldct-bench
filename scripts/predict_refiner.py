"""Denoise test sets with the refiner: CT-Mamba (Stage A), then the refiner (Stage B).

    python scripts/predict_refiner.py --run ct_mamba_refiner --sets mayo_L506 lits [--live-check 3]

Stage A is CT-Mamba's own prediction of the same slice (runs/ct_mamba/predictions/<set>,
made by scripts/predict.py with the frozen CT-Mamba weights), so running CT-Mamba again
would give the identical image. --live-check N re-runs CT-Mamba on N slices per set and
reports the largest difference to the stored Stage A (proof of that). The refiner sees only
images: no masks, no nnU-Net. Output like scripts/predict.py: runs/<run>/predictions/<set>/
<patient>/<name>.npy (float32 HU) + timing.json (refiner time; CT-Mamba time added from
its own timing.json, so seconds_per_slice is the time of the whole two-stage model).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import load_config, resolve   # noqa: E402
from ldct.data import hu_to_unit, load_input, neighbours, test_set_pairs, unit_to_hu   # noqa: E402
from ldct.infer import denoise_slice   # noqa: E402
from ldct.models import build_model   # noqa: E402
from ldct.runtime import device_name, synchronize   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ct_mamba_refiner")
    ap.add_argument("--sets", nargs="+", default=["mayo_L506", "lits"])
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--stage-a-run", default="ct_mamba")
    ap.add_argument("--live-check", type=int, default=0)
    ap.add_argument("--pred-root", help="write to <pred-root>/<set> instead of runs/<run>/predictions/<set>")
    ap.add_argument("--stage-a-root", help="read Stage A from <stage-a-root>/<set> instead of runs/<stage-a-run>/predictions/<set>")
    ap.add_argument("--ct-mamba-checkpoint", default="runs/ct_mamba/best_from11.pth")
    args = ap.parse_args()
    cfg = load_config("ct_mamba")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(resolve(os.path.join("runs", args.run, args.checkpoint)), map_location=device)
    margs = ck.get("model_args") or {}
    model = build_model("ct_mamba_refiner", **margs).to(device)
    context = (int(margs.get("in_channels", 2)) - 2) // 2      # 2.5D: in_channels = 2 * context + 2
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"refiner {args.run}/{args.checkpoint} (iteration {ck.get('iteration')}, loss {ck.get('train_loss')})", flush=True)
    mamba = None
    for set_name in args.sets:
        pairs = test_set_pairs(cfg["data"]["test_sets"][set_name])
        a_dir = os.path.join(resolve(args.stage_a_root), set_name) if args.stage_a_root else resolve(os.path.join("runs", args.stage_a_run, "predictions", set_name))
        out_dir = os.path.join(resolve(args.pred_root), set_name) if args.pred_root else resolve(os.path.join("runs", args.run, "predictions", set_name))
        if args.live_check:
            if mamba is None:
                mk = torch.load(resolve(args.ct_mamba_checkpoint), map_location=device)
                mamba = build_model("ct_mamba").to(device)
                mamba.load_state_dict(mk["model"]); mamba.eval()
            diffs = []
            for low, _, patient, name in pairs[:: max(1, len(pairs) // args.live_check)][: args.live_check]:
                live = unit_to_hu(denoise_slice(mamba, torch.from_numpy(load_input([low]))[None].to(device), cfg)[0, 0].cpu().numpy())
                diffs.append(float(np.abs(live - np.load(os.path.join(a_dir, patient, f"{name}.npy"))).max()))
            print(f"{set_name}: live CT-Mamba vs stored Stage A, max |diff| per slice (HU): {[round(d, 4) for d in diffs]}", flush=True)
        seconds = []
        inputs = neighbours(pairs, context)
        for (low, _, patient, name), lows in tqdm(list(zip(pairs, inputs)), desc=set_name):
            a = np.load(os.path.join(a_dir, patient, f"{name}.npy")).astype(np.float32)
            x = torch.from_numpy(np.concatenate([load_input(lows), hu_to_unit(a)[None]]))[None].to(device)
            synchronize(device); t = time.perf_counter()
            with torch.no_grad():
                y = model(x)
            synchronize(device); seconds.append(time.perf_counter() - t)
            os.makedirs(os.path.join(out_dir, patient), exist_ok=True)
            np.save(os.path.join(out_dir, patient, f"{name}.npy"), unit_to_hu(y[0, 0].float().cpu().numpy()).astype(np.float32))
        timed = seconds[1:] if len(seconds) > 1 else seconds
        stage_a_t = json.load(open(os.path.join(a_dir, "timing.json")))
        timing = {"slices": len(seconds), "refiner_seconds_per_slice": sum(timed) / len(timed),
                  "stage_a_seconds_per_slice": stage_a_t["seconds_per_slice"], "stage_a_device": stage_a_t["device"],
                  "seconds_per_slice": sum(timed) / len(timed) + stage_a_t["seconds_per_slice"],
                  "device": device_name(device), "epoch": ck.get("iteration"), "checkpoint": args.checkpoint}
        json.dump(timing, open(os.path.join(out_dir, "timing.json"), "w"), indent=2)
        print(f"{set_name}: {len(seconds)} slices -> {out_dir} (refiner {timing['refiner_seconds_per_slice'] * 1000:.1f} ms/slice)", flush=True)


if __name__ == "__main__":
    main()
