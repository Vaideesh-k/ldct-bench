"""Quick checks for ldct/infer.py."""
import os
import sys

import torch
import torch.nn as nn

# let this file import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.infer import denoise_slice, tiled

failed = []


def check(label, ok):
    """Print one result and remember failures (the script exits with code 1 at the end)."""
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


torch.manual_seed(0)

# tiled() with a model that changes nothing must give back the input exactly
for H, W in [(128, 128), (100, 77), (513, 512), (40, 50)]:   # even, odd, one extra row, smaller than a patch
    x = torch.rand(1, 1, H, W)
    y = tiled(nn.Identity(), x, patch=64, stride=32, batch=7)
    check(f"identity {H}x{W}: same shape and exact values", y.shape == x.shape and torch.equal(x, y))


class AddOne(nn.Module):
    """Adds 1 to every pixel, so every pixel shows whether it was processed."""
    def forward(self, x):
        return x + 1


x = torch.rand(1, 1, 130, 97)
check("every pixel processed, edges included", torch.allclose(tiled(AddOne(), x, 64, 32), x + 1))


class Counter(nn.Module):
    """Records the input shape of every call."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Conv2d(1, 1, 1)
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        return self.lin(x)


cfg_whole = {"infer": {"patch": None, "stride": 32, "batch": 64}}
cfg_tiled = {"infer": {"patch": 64, "stride": 32, "batch": 64}}
x = torch.rand(1, 1, 128, 128)

m = Counter()
out = denoise_slice(m, x, cfg_whole)
check("patch null -> whole slice in one call", m.shapes == [(1, 1, 128, 128)])
check("no gradients kept", not out.requires_grad)
check("model back in training mode afterwards", m.training)

m = Counter()
denoise_slice(m, x, cfg_tiled)
check("patch 64 -> 64x64 tiles", all(s[-2:] == (64, 64) for s in m.shapes))


class OwnDenoise(Counter):
    def denoise(self, x):
        return torch.zeros_like(x)


m = OwnDenoise()
check("model.denoise() is used when present", torch.equal(denoise_slice(m, x, cfg_tiled), torch.zeros_like(x)) and not m.shapes)

sys.exit(1 if failed else 0)
