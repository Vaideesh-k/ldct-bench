"""Train one model on the Mayo training patients.

    python scripts/train.py --model redcnn              start a new run
    python scripts/train.py --model redcnn --resume     continue after a crash / stop
    python scripts/train.py --model redcnn --smoke      1 epoch, 5 steps: does it run at all?
    python scripts/train.py --model redcnn --epochs 10  fewer epochs than the config
    python scripts/train.py --model ct_mamba --run ct_mamba_50 --config configs/short50.yaml
                                                        a named run with extra settings

Everything goes to runs/<run>/ (run = the model name unless --run is given;
runs/<run>_smoke/ for --smoke):
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
from ldct.lesion import LesionWeightedL1
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
    ap.add_argument("--config", nargs="+", help="extra yaml file(s) merged on top of the configs, in order")
    ap.add_argument("--run", help="name of the run folder under runs/ (default: the model name)")
    ap.add_argument("--init", help="start from the weights in this checkpoint (fine-tuning; not --resume)")
    args = ap.parse_args()
    args.run = args.run or args.model
    return args


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
        for extra in t.get("extra_train_sets") or []:          # e.g. LiTS train patients (Idea 1)
            train_pairs += list_pairs(resolve(extra["root"]), extra_patients(extra))
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nPrepare the data first: python scripts/prep_mayo.py --src <mayo_raw> --dst {d['mayo_root']}")

    lesion = bool(t.get("lesion_weight")) or float(t.get("tumour_patch_share") or 0) > 0
    train_loader = DataLoader(
        PatchPairs(train_pairs, patch=t["patch"], patch_n=t["patch_n"],
                   context=d.get("context", 0), augment=d.get("augment", False),
                   with_masks=lesion, tumour_share=float(t.get("tumour_patch_share") or 0)),
        batch_size=t["batch_size"], shuffle=True, num_workers=t["workers"],
        collate_fn=collate_patches, worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda")
    val_loader = DataLoader(SlicePairs(val_pairs, context=d.get("context", 0)), batch_size=1, num_workers=t["workers"])
    print(f"Training: {len(train_pairs)} slices from {len(d['train_patients'])} patients | "
          f"validation: {len(val_pairs)} slices of {', '.join(d['val_patients'])}")
    return train_loader, val_loader


def extra_patients(spec):
    """Patients of an extra training set: a list, or one part of a split file.
    The locked LiTS test patients can never be chosen: part must be 'train'."""
    if "patients" in spec:
        return list(spec["patients"])
    import json
    with open(resolve(spec["split"])) as f:
        split = json.load(f)
    if spec.get("part", "train") != "train":
        raise SystemExit("extra_train_sets: only the 'train' part of a split may be used for training")
    prefix = spec.get("prefix", "lits_")
    return [f"{prefix}{k:03d}" for k in split["train"]]


def training_loss(model, inp, target, tumour=None, lesion_loss=None, extra_scale=1.0):
    """Use the model's own training step or loss if it has one, else L1.
    With Idea 1 on (lesion_loss given): plain-L1 models use the lesion-weighted L1 instead;
    models with their own loss get `extra_scale * weighted L1` added."""
    if hasattr(model, "training_step"):
        loss = model.training_step(inp, target)
        if lesion_loss is not None:
            raise SystemExit("lesion_weight is not supported for models with a custom training_step")
        return loss
    pred = model(inp)
    if hasattr(model, "compute_loss"):
        loss = model.compute_loss(pred, target, inp)
        return loss + extra_scale * lesion_loss(pred, target, tumour) if lesion_loss is not None else loss
    return lesion_loss(pred, target, tumour) if lesion_loss is not None else F.l1_loss(pred, target)


def validate(model, loader, cfg, device):
    """Mean absolute error in HU over the validation slices (logged only)."""
    errors = []
    for lo, hi, _, _ in tqdm(loader, desc="  validation", leave=False):
        pred = denoise_slice(model, lo.to(device), cfg)
        errors.append(float((unit_to_hu(pred) - unit_to_hu(hi.to(device))).abs().mean()))
    return sum(errors) / len(errors)


def count_params(model):
    """Parameters of the denoiser itself; modules the model lists in `loss_modules`
    (a loss network, an EMA copy) are counted separately: not trained as the denoiser."""
    loss_parts = [getattr(model, name) for name in getattr(model, "loss_modules", ())]
    loss_ids = {id(p) for m in loss_parts for p in m.parameters()}
    net = sum(p.numel() for p in model.parameters() if id(p) not in loss_ids)
    return net, sum(p.numel() for m in loss_parts for p in m.parameters())


def choose_best(epoch, train_loss, best_loss, best_epoch, best_from=1):
    """The final model is the epoch with the lowest training loss, counting only
    epochs >= best_from. A model whose loss formula changes during training
    (CT-Mamba adds terms at epoch 11) sets best_from there, so best.pth is never
    stuck at an early epoch whose loss used a smaller formula. Earlier epochs still
    update best.pth until then (so a short or smoke run always has one).
    Returns (best_loss, best_epoch, this_epoch_is_new_best)."""
    if best_epoch < best_from <= epoch:
        best_loss = math.inf   # forget bests from before the switch
    if train_loss < best_loss:
        return train_loss, epoch, True
    return best_loss, best_epoch, False


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
        run_dir = os.path.join(resolve(cfg["output_dir"]), f"{args.run}_smoke")
        shutil.rmtree(run_dir, ignore_errors=True)   # a smoke run always starts clean
    else:
        if args.epochs:
            t["epochs"] = args.epochs
        run_dir = os.path.join(resolve(cfg["output_dir"]), args.run)
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
          + (f", plus {loss_net:,} in {'/'.join(getattr(model, 'loss_modules', ()))} (not counted)" if loss_net else "") + f" | device: {device}")

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
    if args.init and start_epoch == 1:
        ckpt = torch.load(resolve(args.init), map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Initialised from {args.init} (epoch {ckpt.get('epoch')}); optimiser and schedule start fresh.")

    if start_epoch > t["epochs"]:
        print(f"Already trained {t['epochs']} epochs. Nothing to do.")
        return

    train_loader, val_loader = make_loaders(cfg, device)
    torch.manual_seed(cfg["seed"] + start_epoch)   # a resumed run does not repeat the same shuffles
    lesion_loss = LesionWeightedL1(t["lesion_weight"]) if t.get("lesion_weight") else None
    if lesion_loss:
        print(f"Idea 1: lesion-weighted L1 (x{t['lesion_weight']} inside tumours, extra scale "
              f"{t.get('lesion_extra_scale', 1.0)}), tumour patch share {t.get('tumour_patch_share') or 0}")

    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        model.epoch = epoch   # lets a loss change during training (see template.py)
        start, total, steps = time.time(), 0.0, 0
        lr = optimizer.param_groups[0]["lr"]

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
            if t["max_steps_per_epoch"] and steps >= t["max_steps_per_epoch"]:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            tumour = batch[2].to(device) if len(batch) == 3 else None
            loss = training_loss(model, x, y, tumour, lesion_loss, float(t.get("lesion_extra_scale", 1.0)))
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

        # the final model is the epoch with the LOWEST TRAINING LOSS (as in CT-Mamba),
        # among epochs >= train.best_from_epoch
        if epoch == t["best_from_epoch"] > 1:
            print(f"From epoch {epoch} on, best.pth is chosen only among epochs >= {epoch}.")
        best_loss, best_epoch, improved = choose_best(epoch, train_loss, best_loss, best_epoch,
                                                      t["best_from_epoch"])
        if improved:
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
