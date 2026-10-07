"""Quick checks for ldct/metrics.py."""
import os
import sys

import numpy as np

# let this file import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.metrics import score

failed = []


def check(label, ok):
    """Print one result and remember failures (the script exits with code 1 at the end)."""
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


rng = np.random.default_rng(0)
clean = rng.uniform(-100, 200, (128, 128))   # a fake slice inside the window

# identical images: perfect scores
s = score(clean, clean)
check("identical: rmse is 0", s["rmse"] == 0.0)
check("identical: ssim is 1", abs(s["ssim"] - 1) < 1e-9)

# more noise -> lower PSNR
small = score(clean + rng.normal(0, 5, clean.shape), clean)
big = score(clean + rng.normal(0, 50, clean.shape), clean)
print(f"psnr: small noise {small['psnr']:.1f} dB, big noise {big['psnr']:.1f} dB")
check("added noise lowers psnr", small["psnr"] > big["psnr"])

# values outside the window are clipped: bone at 1000 vs 2000 HU counts as equal
check("outside window ignored", score(np.full((64, 64), 1000.0), np.full((64, 64), 2000.0))["rmse"] == 0.0)

# shape mismatch gives a clear error
try:
    score(clean, clean[:64])
    check("caught shape mismatch", False)
except ValueError:
    check("caught shape mismatch", True)

sys.exit(1 if failed else 0)
