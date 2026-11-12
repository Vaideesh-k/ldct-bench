"""Watch a running CT-Mamba training job (CPU only) and keep the weights that
train.py's best.pth rule would miss, without touching the training itself.

Every POLL seconds it reads runs/ct_mamba/train_log.csv. For each newly finished
epoch N (the log line is written only after latest.pth is complete):
  - if N >= FROM and N has the lowest train_loss among epochs >= FROM:
        save its weights as best_from11.pth (+ best_from11.json)
  - if N is a multiple of EVERY: save its weights as epoch_<N>.pth
Both are model-only, in the same format as best.pth, so
    python scripts/predict.py --model ct_mamba --checkpoint best_from11.pth
works. It only reads latest.pth and only writes its own files.

    python slurm/watch_best.py            (run by slurm/watch_best.sbatch)
"""
import csv
import datetime
import json
import os
import shutil
import time

import torch

RUN = os.environ.get("WATCH_RUN", os.path.expanduser("~/ldct-bench/runs/ct_mamba"))
FROM = 11          # CT-Mamba's loss gains 2 terms at epoch 11
EVERY = int(os.environ.get("WATCH_EVERY", 25))  # also keep epoch_25.pth, epoch_50.pth, ...
LAST = int(os.environ.get("WATCH_LAST", 250))   # stop after this epoch
POLL = float(os.environ.get("WATCH_POLL", 60))  # seconds; an epoch takes 7-15 min

LOG = os.path.join(RUN, "train_log.csv")
LATEST = os.path.join(RUN, "latest.pth")
BEST = os.path.join(RUN, "best_from11.pth")
INFO = os.path.join(RUN, "best_from11.json")
TMP_COPY = os.path.join(RUN, ".watch_latest_copy.tmp")


def say(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def read_log():
    """{epoch: train_loss} of every finished epoch."""
    if not os.path.exists(LOG):
        return {}
    with open(LOG, newline="") as f:
        return {int(r["epoch"]): float(r["train_loss"]) for r in csv.DictReader(f)}


def weights_of_epoch(n):
    """The model weights of epoch n, read safely from latest.pth, or None if
    training has already replaced latest.pth with a newer epoch.

    train.py writes latest.pth.tmp and renames it (atomic), so an open file is
    always one complete checkpoint: we copy through the open handle (a rename
    during the copy cannot mix two epochs), then check the epoch stored inside."""
    with open(LATEST, "rb") as src, open(TMP_COPY, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
    try:
        ckpt = torch.load(TMP_COPY, map_location="cpu", mmap=True)   # mmap: low RAM use
        if ckpt["epoch"] != n:
            say(f"epoch {n}: latest.pth already holds epoch {ckpt['epoch']}, missed it")
            return None
        return ckpt["model"]
    finally:
        os.remove(TMP_COPY)


def save_weights(model, n, loss, path):
    """Same format as best.pth; written to .tmp and renamed, so never half-written."""
    torch.save({"model": model, "epoch": n, "train_loss": loss, "model_args": {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def disk_used_gb():
    total = 0
    for dirpath, _, files in os.walk(os.environ.get("WATCH_HOME", os.path.expanduser("~"))):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total / 1e9


def main():
    torch.set_num_threads(1)
    info = json.load(open(INFO)) if os.path.exists(INFO) else {"epoch": None, "train_loss": None, "missed": []}
    losses = read_log()
    seen = max(losses, default=0)
    say(f"start: {RUN} | {len(losses)} epochs in the log so far (last: {seen}) | "
        f"best_from11 so far: epoch {info['epoch']} | home disk {disk_used_gb():.1f} GB of 50")
    for e in sorted(losses):
        say(f"  epoch {e}: train_loss {losses[e]}")

    # the newest epoch at start-up can still be captured (latest.pth holds it)
    pending = [seen] if seen else []
    while True:
        losses = read_log()
        newest = max(losses, default=0)
        pending += list(range(seen + 1, newest + 1))
        seen = max(seen, newest)

        for n in pending:
            later = {e: l for e, l in losses.items() if e >= FROM}
            is_best = n >= FROM and losses[n] == min(later.values()) and n != info["epoch"]
            snapshot = n % EVERY == 0 and not os.path.exists(os.path.join(RUN, f"epoch_{n}.pth"))
            if not (is_best or snapshot):
                continue
            model = weights_of_epoch(n) if n == newest else None
            if model is None:
                if is_best:
                    info["missed"].append(n)
                    json.dump(info, open(INFO, "w"), indent=1)
                say(f"epoch {n}: could not capture (no longer in latest.pth)")
                continue
            if is_best:
                save_weights(model, n, losses[n], BEST)
                info.update(epoch=n, train_loss=losses[n],
                            rule=f"lowest train_loss among epochs >= {FROM}",
                            saved=datetime.datetime.now().isoformat(timespec="seconds"))
                json.dump(info, open(INFO, "w"), indent=1)
                say(f"epoch {n}: new best from epoch {FROM} on (loss {losses[n]}) -> best_from11.pth")
            if snapshot:
                save_weights(model, n, losses[n], os.path.join(RUN, f"epoch_{n}.pth"))
                say(f"epoch {n}: snapshot -> epoch_{n}.pth | home disk {disk_used_gb():.1f} GB of 50")
            del model
        pending = []

        if seen >= LAST:
            say(f"epoch {LAST} reached: done. best_from11 = epoch {info['epoch']}")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
