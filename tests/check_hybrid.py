"""Checks for ldct/models/ct_mamba_hybrid.py.
CPU part (any machine): the attention block alone.
GPU part (NVIDIA + mamba-ssm, e.g. a MIG slice): the 4 full models - shapes, parameter
counts, peak memory at the official batch of 32 patches, seconds per training step."""
import argparse
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--only", help="GPU part: run only this model (plain | low | small | parallel); one process per model avoids leftover memory")
ap.add_argument("--batches", default="32,16,8", help="batch sizes to try in turn until one fits")
ARGS = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.models.ct_mamba_hybrid import T2TDilatedAttention, ParallelBlock, VARIANTS   # noqa: E402

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- CPU: the block itself
for dim, grid in ([] if ARGS.only else ((32, 32), (128, 16), (512, 8), (96, 32))):
    blk = T2TDilatedAttention(dim, grid)
    x = torch.randn(2, grid, grid, dim)
    y = blk(x)
    check(f"attention block dim {dim} grid {grid}: (B,H,W,C) in and out, {n_params(blk):,} params", y.shape == x.shape)
blk = T2TDilatedAttention(32, 32).eval()
x = torch.randn(1, 32, 32, 32)
check("residual: forward = x + inner", torch.allclose(blk(x), x + blk.inner(x), atol=1e-5))
check("dilated unfold keeps the grid (padding = dilation)", blk.unfold(x.permute(0, 3, 1, 2)).shape == (1, 9 * 32, 32 * 32))
y = blk(x)
check("eval is deterministic", torch.allclose(blk(x), y))
x2 = x.clone(); x2[0, 5, 5, :] += 1.0
d = (blk(x2) - y).abs().sum(-1)[0]
check("one changed token changes every output token (global attention)", bool((d > 1e-6).float().mean() > 0.9))
try:
    T2TDilatedAttention(30, 8); check("dim not divisible by heads raises", False)
except ValueError:
    check("dim not divisible by heads raises", True)


class FakeMamba(torch.nn.Module):      # stands in for a CZSS block on the CPU
    def forward(self, x):
        return x * 3.0


pb = ParallelBlock(FakeMamba(), T2TDilatedAttention(32, 8)).eval()
x = torch.randn(1, 8, 8, 32)
expect = x + 0.5 * (x * 3.0 - x) + 0.5 * pb.attention.inner(x)
check("parallel block: gate starts at 0.5 and mixes residual updates", torch.allclose(pb(x), expect, atol=1e-5))
pb.gate.data.fill_(-20.0)
check("parallel block with gate -> 0 equals the Mamba block", torch.allclose(pb(x), x * 3.0, atol=1e-4))

# ---------------------------------------------------------------- GPU: the full models
if not torch.cuda.is_available():
    print("\nno CUDA: skipped the full-model checks (run slurm/hybrid_test.sbatch on the DGX)")
elif not ARGS.only:
    print("\nCUDA found: run the GPU part one model at a time: --only plain|low|small|parallel")
else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train import count_params, training_loss
    from ldct.models import build_model
    dev = torch.device("cuda")
    label = ARGS.only
    name, kwargs = ("ct_mamba", {}) if label == "plain" else ("ct_mamba_hybrid", {"variant": label})
    torch.manual_seed(0)
    model = build_model(name, **kwargs).to(dev).train()
    model.epoch = 11                                   # full Deep-NPS loss, the expensive case
    net, _ = count_params(model)
    with torch.no_grad():
        model.eval(); out = model(torch.rand(4, 1, 64, 64, device=dev)); model.train()
    check(f"{label}: (4,1,64,64) -> same shape", out.shape == (4, 1, 64, 64))
    if label != "plain":
        check(f"{label}: swapped 4 block(s)", model.swapped == 4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    result = None
    for bs in [int(b) for b in ARGS.batches.split(",")]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.rand(bs, 1, 64, 64, device=dev); y = torch.rand(bs, 1, 64, 64, device=dev)
            times = []
            for i in range(8):
                torch.cuda.synchronize(); t0 = time.time()
                loss = training_loss(model, x, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize(); times.append(time.time() - t0)
            check(f"{label}: finite loss after 8 steps at batch {bs}", torch.isfinite(loss).item())
            result = (bs, sum(times[3:]) / len(times[3:]), torch.cuda.max_memory_allocated() / 2**30)
            break
        except torch.OutOfMemoryError:
            print(f"  {label}: batch {bs} does not fit this GPU slice, trying smaller")
            opt.zero_grad(set_to_none=True)
    check(f"{label}: trained at some batch size", result is not None)
    if result:
        bs, step, peak = result
        scale = 32 / bs
        extra = f" | scaled to batch 32: ~{step * scale:.2f} s/step, ~{peak * scale:.1f} GiB (estimate)" if bs != 32 else ""
        print(f"RESULT | {label} | {net:,} params | batch {bs} | {step:.3f} s/step | peak {peak:.2f} GiB{extra}")
    if label == "parallel":
        print(f"  gates: {model.gates()}")

print("\nFAILED:" if failed else "\nAll hybrid checks passed.", failed or "")
sys.exit(1 if failed else 0)
