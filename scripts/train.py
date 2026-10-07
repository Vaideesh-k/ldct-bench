"""Train one model on the Mayo training patients.

    python scripts/train.py --model redcnn              start a new run
    python scripts/train.py --model redcnn --resume     continue after a crash / stop
    python scripts/train.py --model redcnn --smoke      1 epoch, 5 steps: does it run at all?
    python scripts/train.py --model redcnn --epochs 10  fewer epochs than the config

Everything goes to runs/<model>/ (runs/<model>_smoke/ for --smoke):
    latest.pth      saved every epoch, used by --resume
    best.pth        the epoch with the lowest training loss = the final model
    train_log.csv   one row per epoch: epoch, train_loss, val_l1, lr, seconds
Validation (every val_every epochs) is only logged; it never picks the model.
"""
import argparse
import csv
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# let this script import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import load_config, resolve
from ldct.data import PatchPairs, SlicePairs, collate_patches, list_pairs, unit_to_hu
from ldct.infer import denoise_slice
from ldct.models import build_model
from ldct.runtime import pick_device

LOG_COLUMNS = ["epoch", "train_loss", "val_l1", "lr", "seconds"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="name of a file in ldct/models/, e.g. redcnn")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<model>/latest.pth")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 5 steps, saved in runs/<model>_smoke")
    ap.add_argument("--epochs", type=int, help="override train.epochs from the config")
    ap.add_argument("--config", help="extra yaml merged on top of the configs (used by tests)")
    return ap.parse_args()


def seed_worker(worker_id):
    """Give every data-loading process its own random patches."""
    np.random.seed(torch.initial_seed() % 2 ** 32)


def make_loaders(cfg, device):
    """Training: random patch pairs. Validation: every k-th whole slice."""
    d, t = cfg["data"], cfg["train"]
    root = resolve(d["mayo_root"])
    try:
        train_pairs = list_pairs(root, d["train_patients"])
        val_pairs = list_pairs(root, d["val_patients"], every=d["val_every_kth_slice"])
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nPrepare the data first: python scripts/prep_mayo.py --src <mayo_raw> --dst {d['mayo_root']}")

    train_loader = DataLoader(
        PatchPairs(train_pairs, patch=t["patch"], patch_n=t["patch_n"]),
        batch_size=t["batch_size"], shuffle=True, num_workers=t["workers"],
        collate_fn=collate_patches, worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda")
    val_loader = DataLoader(SlicePairs(val_pairs), batch_size=1, num_workers=t["workers"])
    print(f"Training: {len(train_pairs)} slices from {len(d['train_patients'])} patients | "
          f"validation: {len(val_pairs)} slices of {', '.join(d['val_patients'])}")
    return train_loader, val_loader


def training_loss(model, inp, target):
    """Use the model's own training step or loss if it has one, else L1."""
    if hasattr(model, "training_step"):
        return model.training_step(inp, target)
    pred = model(inp)
    if hasattr(model, "compute_loss"):
        return model.compute_loss(pred, target, inp)
    return F.l1_loss(pred, target)


def validate(model, loader, cfg, device):
    """Mean absolute error in HU over the validation slices (logged only)."""
    errors = []
    for lo, hi, _, _ in tqdm(loader, desc="  validation", leave=False):
        pred = denoise_slice(model, lo.to(device), cfg)
        errors.append(float((unit_to_hu(pred) - unit_to_hu(hi.to(device))).abs().mean()))
    return sum(errors) / len(errors)


def count_params(model):
    """Parameters of the denoiser itself; a loss network (if the model lists it
    in `loss_modules`) is counted separately because it is not used at test time."""
    loss_parts = [getattr(model, name) for name in getattr(model, "loss_modules", ())]
    loss_ids = {id(p) for m in loss_parts for p in m.parameters()}
    net = sum(p.numel() for p in model.parameters() if id(p) not in loss_ids)
    return net, sum(p.numel() for m in loss_parts for p in m.parameters())


def save(state, path):
    """Write to a temporary file first, so a crash mid-save never corrupts the checkpoint."""
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def append_log(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    try:
        cfg = load_config(args.model, args.config)
    except ValueError as e:   # L506 in the training or validation patients
        raise SystemExit(f"Refusing to train: {e}")
    t = cfg["train"]

    if args.smoke:
        t["epochs"], t["max_steps_per_epoch"] = 1, 5
        run_dir = os.path.join(resolve(cfg["output_dir"]), f"{args.model}_smoke")
        shutil.rmtree(run_dir, ignore_errors=True)   # a smoke run always starts clean
    else:
        if args.epochs:
            t["epochs"] = args.epochs
        run_dir = os.path.join(resolve(cfg["output_dir"]), args.model)
    os.makedirs(run_dir, exist_ok=True)
    latest, best, log = (os.path.join(run_dir, f) for f in ("latest.pth", "best.pth", "train_log.csv"))

    if os.path.exists(latest) and not args.resume and not args.smoke:
        raise SystemExit(f"{run_dir} already has a run. Use --resume to continue it, "
                         f"or move/delete that folder to start over.")

    device = pick_device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = build_model(args.model, **cfg["model_args"]).to(device)
    net, loss_net = count_params(model)
    print(f"Model {args.model}: {net:,} parameters ({net / 1e6:.2f} M)"
          + (f", plus {loss_net:,} in its loss network" if loss_net else "") + f" | device: {device}")

    # only parameters that are meant to learn (a frozen pretrained loss network is skipped)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=t["lr"], betas=tuple(t["betas"]), weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"], eta_min=t["lr_min"])

    start_epoch, best_loss, best_epoch = 1, math.inf, 0
    if args.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scheduler.T_max = t["epochs"]   # --epochs may have changed since the run started
        start_epoch, best_loss, best_epoch = ckpt["epoch"] + 1, ckpt["best_loss"], ckpt["best_epoch"]
        print(f"Resuming after epoch {ckpt['epoch']} (best so far: epoch {best_epoch}, loss {best_loss:.6f})")
    elif args.resume:
        print(f"No {latest} yet: starting a new run.")

    if start_epoch > t["epochs"]:
        print(f"Already trained {t['epochs']} epochs. Nothing to do.")
        return

    train_loader, val_loader = make_loaders(cfg, device)
    torch.manual_seed(cfg["seed"] + start_epoch)   # a resumed run does not repeat the same shuffles

    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        model.epoch = epoch   # lets a loss change during training (see template.py)
        start, total, steps = time.time(), 0.0, 0
        lr = optimizer.param_groups[0]["lr"]

        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
            if t["max_steps_per_epoch"] and steps >= t["max_steps_per_epoch"]:
                break
            loss = training_loss(model, x.to(device), y.to(device))
            if not torch.isfinite(loss):
                raise SystemExit(f"The loss became {loss.item()} at epoch {epoch}. "
                                 f"latest.pth still holds epoch {epoch - 1}; try a lower lr.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
        scheduler.step()
        train_loss = total / steps

        val_l1 = ""
        if epoch % t["val_every"] == 0 or epoch == t["epochs"]:
            val_l1 = round(validate(model, val_loader, cfg, device), 4)

        # the final model is the epoch with the LOWEST TRAINING LOSS (as in CT-Mamba)
        if train_loss < best_loss:
            best_loss, best_epoch = train_loss, epoch
            save({"model": model.state_dict(), "epoch": epoch, "train_loss": train_loss,
                  "model_args": cfg["model_args"]}, best)
        save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "epoch": epoch,
              "best_loss": best_loss, "best_epoch": best_epoch}, latest)

        seconds = round(time.time() - start, 1)
        append_log(log, {"epoch": epoch, "train_loss": round(train_loss, 6), "val_l1": val_l1,
                         "lr": f"{lr:.3g}", "seconds": seconds})
        print(f"epoch {epoch}: train_loss {train_loss:.6f} | val_l1 {val_l1 or '-'} HU | "
              f"lr {lr:.3g} | {seconds}s | best epoch {best_epoch}")

    print(f"Done. Final model: {best} (epoch {best_epoch}, training loss {best_loss:.6f})")


if __name__ == "__main__":
    main()
