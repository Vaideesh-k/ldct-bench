"""Time training steps of one model on this GPU and estimate the full run.

    python slurm/bench_step.py --model ctformer [--steps 30] [--batch 4]

--batch overrides train.batch_size (slices per step) when the real batch does not
fit the GPU slice used for the test; the memory and time per sample are then
scaled to the real batch in the printout (an estimate, marked as such).
No checkpoints are written.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from train import count_params, make_loaders, training_loss, validate   # noqa: E402
from ldct.config import load_config                                      # noqa: E402
from ldct.models import build_model                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--warm", type=int, default=5)
ap.add_argument("--batch", type=int, help="override train.batch_size for this test")
ap.add_argument("--no-val", action="store_true")
args = ap.parse_args()

cfg = load_config(args.model)
t = cfg["train"]
real_batch = t["batch_size"]
if args.batch:
    t["batch_size"] = args.batch
t["workers"] = min(t["workers"], 8)
dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

torch.cuda.reset_peak_memory_stats()
model = build_model(args.model, **cfg["model_args"]).to(dev)
model.train()
net, extra = count_params(model)
print(f"model {args.model}: {net:,} parameters" + (f" (+{extra:,} not trained)" if extra else ""))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t["lr"], weight_decay=t["weight_decay"])
tl, vl = make_loaders(cfg, dev)
steps_per_epoch_real = -(-len(tl.dataset) // real_batch)
print(f"steps per epoch at the real batch {real_batch}: {steps_per_epoch_real} | epochs: {t['epochs']}")
it = iter(tl)


def run(epoch, n, warm):
    model.epoch = epoch
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize()
            start = time.time()
        try:
            x, y = next(it)
        except StopIteration:
            break
        if i == 0:
            print("batch:", tuple(x.shape), "->", tuple(y.shape))
        loss = training_loss(model, x.to(dev), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - start) / n


sec = run(1, args.steps, args.warm)
peak = torch.cuda.max_memory_allocated() / 2**30
scale = real_batch / t["batch_size"]
print(f"sec/step: {sec:.3f} at batch {t['batch_size']} | peak GPU memory {peak:.2f} GiB"
      + (f" | scaled to batch {real_batch}: ~{sec * scale:.3f} s/step, ~{peak * scale:.1f} GiB (estimate)" if scale != 1 else ""))
v = 0.0
if not args.no_val:
    start = time.time()
    validate(model, vl, cfg, dev)
    v = time.time() - start
    print(f"validation pass: {v:.1f}s ({len(vl)} slices) -> {v / len(vl):.2f} s/slice")
epoch_s = steps_per_epoch_real * sec * scale
n_val = t["epochs"] // t["val_every"]
total = t["epochs"] * epoch_s + n_val * v
print(f"epoch ~{epoch_s / 60:.1f} min; {t['epochs']} epochs + {n_val} validations ~{total / 3600:.1f} h = {total / 86400:.2f} days")
print("BENCH_OK")
