"""Step 2: frozen CT-Mamba (Stage A) outputs for the refiner's TRAINING slices, saved as int16 HU
next to the inputs: <root>/<patient>/ct_mamba/<name>.npy.

    python scripts/precompute_stage_a.py --sets mayo_train lits_train [--checkpoint runs/ct_mamba/best_from11.pth]

mayo_train = the 9 Mayo training patients (never L506); lits_train = the train part of
configs/lits_split.json (the locked test patients are refused). The test patients' CT-Mamba
outputs are the normal CT-Mamba predictions (runs/ct_mamba/predictions/<set>), made with the
same weights. Same tiled inference as scripts/predict.py (64x64 tiles, stride 32, averaged).
Resumable (existing files are skipped). Stops before the home folder passes --max-used-gb.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import load_config, resolve   # noqa: E402
from ldct.data import list_pairs, load_input, unit_to_hu   # noqa: E402
from ldct.infer import denoise_slice   # noqa: E402
from ldct.models import build_model   # noqa: E402
from ldct.tumour_train import lits_train_patients, stage_a_path   # noqa: E402


def used_gb():
    out = subprocess.run(["du", "-s", "--block-size=1M", os.path.expanduser("~")], text=True, capture_output=True).stdout
    return int(out.split()[0]) / 1024


def to_int16(hu):
    return np.clip(np.rint(hu), -32768, 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["lits_train", "mayo_train"])
    ap.add_argument("--checkpoint", default="runs/ct_mamba/best_from11.pth")
    ap.add_argument("--tile-batch", type=int, default=256)
    ap.add_argument("--max-used-gb", type=float, default=39.5)
    args = ap.parse_args()
    cfg = load_config("ct_mamba")
    cfg["infer"]["batch"] = args.tile_batch
    split = json.load(open(resolve("configs/lits_split.json")))
    jobs = []
    if "mayo_train" in args.sets:
        pats = cfg["data"]["train_patients"]
        assert "L506" not in pats
        jobs += list_pairs(resolve("data/mayo"), pats)
    if "lits_train" in args.sets:
        pats = lits_train_patients(resolve("configs/lits_split.json"))
        assert not {f"lits_{k:03d}" for k in split["test"]} & set(pats)
        jobs += list_pairs(resolve("data/lits"), pats)
    todo = [p for p in jobs if not os.path.exists(stage_a_path(p[0]))]
    print(f"{len(jobs)} slices, {len(todo)} to do", flush=True)
    device = torch.device("cuda")
    ck = torch.load(resolve(args.checkpoint), map_location=device)
    model = build_model("ct_mamba").to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"CT-Mamba {args.checkpoint} (epoch {ck.get('epoch')})", flush=True)
    t0 = time.time()
    for n, (low, _, patient, name) in enumerate(todo):
        if n % 500 == 0:
            gb = used_gb()
            print(f"{n}/{len(todo)} slices, {time.time() - t0:.0f}s, home {gb:.1f} GB", flush=True)
            if gb > args.max_used_gb:
                raise SystemExit(f"STOP: home uses {gb:.1f} GB > {args.max_used_gb}; resubmit after freeing space")
        x = torch.from_numpy(load_input([low]))[None].to(device)
        out = unit_to_hu(denoise_slice(model, x, cfg)[0, 0].float().cpu().numpy())
        dst = stage_a_path(low)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst + ".tmp.npy", to_int16(out))
        os.replace(dst + ".tmp.npy", dst)
    print(f"done: {len(todo)} slices in {time.time() - t0:.0f}s, home {used_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
