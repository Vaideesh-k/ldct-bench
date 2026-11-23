"""Train the REFINER (step 3) or its CONTROL (step 4) with the shared data/loss of ldct/tumour_train.py.

    python scripts/train_tumour.py --mode refiner [--run ct_mamba_refiner] [--iterations N] [--smoke]
    python scripts/train_tumour.py --mode control [--run ct_mamba_control]  [--iterations N] [--smoke]

refiner: ct_mamba_refiner (2D U-Net residual on top of the frozen, precomputed CT-Mamba output)
control: CT-Mamba itself, initialised from runs/ct_mamba/best_from11.pth, fine-tuned with the
         same data mix, sampling and loss (L1 + lesion-weighted L1). CT-Mamba only accepts
         64x64 inputs, so it gets 64 patches of 64^2 per step = the refiner's 16 x 128^2 pixels.
Iteration-based: AdamW, cosine lr over all iterations, the loss averaged over windows of
`window` iterations; best.pth = the window with the lowest training loss (fixed rule),
latest.pth every window (resume with --resume). runs/<run>/train_log.csv: one row per window.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import resolve   # noqa: E402
from ldct.models import build_model   # noqa: E402
from ldct.tumour_train import FromIndex, MixedPatches, gradient_lesion, l1_lesion, lits_train_patients   # noqa: E402

LOG = ["iteration", "window_loss", "loss_mayo", "loss_lits", "lits_patches_with_tumour", "lr", "seconds", "edge_term"]


def load_cfg(path):
    return yaml.safe_load(open(resolve(path)))


def build(mode, cfg, device):
    m = cfg["modes"][mode]
    model = build_model(m["model"], **m.get("model_args", {})).to(device)
    if m.get("init"):
        ck = torch.load(resolve(m["init"]), map_location=device)
        model.load_state_dict(ck["model"])
        print(f"initialised from {m['init']} (epoch {ck.get('epoch')})", flush=True)
    return model


def trainable(model, mode):
    """The refiner trains everything. The control trains only CT-Mamba's denoiser: its
    loss-side modules (noise feature nets, ResNet-50) are not used by this loss."""
    skip = {id(p) for name in getattr(model, "loss_modules", ()) for p in getattr(model, name).parameters()}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in skip]


def save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["refiner", "refiner25d", "control", "edge"])
    ap.add_argument("--config", default="configs/tumour_train.yaml")
    ap.add_argument("--run")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="40 iterations, window 10, in runs/<run>_smoke")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    m = cfg["modes"][args.mode]
    run = args.run or f"ct_mamba_{args.mode}"
    iters, window = args.iterations or cfg["iterations"], cfg["window"]
    if args.smoke:
        run, iters, window = run + "_smoke", 40, 10
    run_dir = resolve(os.path.join("runs", run))
    os.makedirs(run_dir, exist_ok=True)
    latest, best, log = (os.path.join(run_dir, f) for f in ("latest.pth", "best.pth", "train_log.csv"))
    if os.path.exists(latest) and not args.resume and not args.smoke:
        raise SystemExit(f"{run_dir} has a run; use --resume")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg["seed"])

    import json
    split = json.load(open(resolve(cfg["lits_split"])))
    data = MixedPatches(resolve(cfg["mayo_root"]), cfg["mayo_patients"], resolve(cfg["lits_root"]),
                        lits_train_patients(resolve(cfg["lits_split"])), patch=m["patch"],
                        n_items=iters * m["batch"], tumour_share=cfg["tumour_share"],
                        stage_a=m["stage_a"], dilate=cfg["dilate"], seed=cfg["seed"],
                        locked=[f"lits_{k:03d}" for k in split["test"]], context=int(m.get("context", 0)))
    print(f"mode {args.mode}: model {m['model']}, {m['batch']} x {m['patch']}^2 patches/step "
          f"({m['batch'] * m['patch'] ** 2:,} px), {iters} iterations, "
          f"{len(data.mayo)} Mayo + {len(data.lits)} LiTS train slices ({len(data.tumour_slices)} with tumour)", flush=True)

    model = build(args.mode, cfg, device)
    params = trainable(model, args.mode)
    print(f"trainable parameters: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=cfg["lr_min"])
    start, best_loss = 0, math.inf
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start, best_loss = ck["iteration"], ck["best_loss"]
        print(f"resuming after iteration {start} (best window loss {best_loss:.6f})", flush=True)

    edge_w = float(m.get("edge_weight", 0.0))

    def objective(pred, y, mask):   # control/refiner: L1 + lesion L1; edge: + edge_w * gradient term
        loss = l1_lesion(pred, y, mask, cfg["lesion_weight"])
        if not edge_w:
            return loss, 0.0
        g = gradient_lesion(pred, y, mask, cfg["lesion_weight"])
        return loss + edge_w * g, float(g.detach())

    accum = int(m.get("accum", 1))
    while True:   # find a micro-batch that fits: the step is always m["batch"] patches
        try:
            micro = m["batch"] // accum
            assert micro * accum == m["batch"], "batch must be divisible by accum"
            loader = DataLoader(data, batch_size=micro, sampler=FromIndex(start * m["batch"], iters * m["batch"]),
                                num_workers=cfg["workers"], pin_memory=device.type == "cuda", drop_last=True)
            probe = next(iter(loader))
            model.train()
            out = model(probe[0].to(device))
            objective(out, probe[1].to(device), probe[2].to(device))[0].backward()
            opt.zero_grad(set_to_none=True)
            del out, probe
            break
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            accum *= 2
            print(f"out of memory: gradient accumulation over {accum} micro-batches of {m['batch'] // accum}", flush=True)
            if accum > m["batch"]:
                raise
    print(f"micro-batch {micro} x accumulation {accum} = {m['batch']} patches per optimizer step", flush=True)
    model.train()
    it, t0 = start, time.time()
    acc = {"loss": 0.0, "mayo": 0.0, "lits": 0.0, "n_mayo": 0, "n_lits": 0, "tum": 0.0, "lits_px": 0.0, "n": 0, "edge": 0.0}
    step_loss, k = 0.0, 0
    for x, y, mask, src in loader:
        x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        pred = model(x)
        loss, edge = objective(pred, y, mask)
        acc["edge"] += edge / accum
        if not torch.isfinite(loss):
            raise SystemExit(f"loss {loss.item()} at iteration {it + 1}; latest.pth holds the last window")
        (loss / accum).backward()
        step_loss += loss.item() / accum
        k += 1
        with torch.no_grad():   # per-source plain L1 for the log
            e = (pred - y).abs().mean(dim=(1, 2, 3))
            lits = src.to(device) == 1
            acc["mayo"] += float(e[~lits].sum()); acc["n_mayo"] += int((~lits).sum())
            acc["lits"] += float(e[lits].sum()); acc["n_lits"] += int(lits.sum())
            acc["tum"] += float(mask[lits].any(dim=(1, 2, 3)).float().sum())
        if k < accum:
            continue
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        loss_value, step_loss, k = step_loss, 0.0, 0
        it += 1
        acc["loss"] += loss_value; acc["n"] += 1
        if it % window == 0 or it == iters:
            wl = acc["loss"] / acc["n"]
            row = {"iteration": it, "window_loss": round(wl, 7),
                   "loss_mayo": round(acc["mayo"] / max(acc["n_mayo"], 1), 7),
                   "loss_lits": round(acc["lits"] / max(acc["n_lits"], 1), 7),
                   "lits_patches_with_tumour": round(acc["tum"] / max(acc["n_lits"], 1), 4),   # LiTS patches with tumour
                   "lr": f"{sched.get_last_lr()[0]:.3g}", "seconds": round(time.time() - t0, 1),
                   "edge_term": round(acc["edge"] / acc["n"], 7)}
            new = not os.path.exists(log)
            with open(log, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG)
                if new:
                    w.writeheader()
                w.writerow(row)
            if wl < best_loss:
                best_loss = wl
                save({"model": model.state_dict(), "epoch": it, "iteration": it, "train_loss": wl,
                      "model_args": m.get("model_args", {}), "mode": args.mode}, best)
            save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                  "iteration": it, "best_loss": best_loss}, latest)
            print(f"it {it}/{iters}: window loss {wl:.6f} (Mayo L1 {row['loss_mayo']:.6f}, LiTS L1 {row['loss_lits']:.6f}, "
                  f"LiTS patches with tumour {row['lits_patches_with_tumour']:.2f}) lr {row['lr']} | "
                  f"{row['seconds']:.0f}s | best {best_loss:.6f}", flush=True)
            acc = dict.fromkeys(acc, 0)
            acc.update({"loss": 0.0, "mayo": 0.0, "lits": 0.0, "tum": 0.0, "edge": 0.0})
            t0 = time.time()
    print(f"Done. best.pth = window loss {best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
