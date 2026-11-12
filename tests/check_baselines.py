"""Checks for the CTformer and CoreDiff wrappers and the pipeline pieces they need
(slice context, patch augmentation, per-slice scores). Small fake data, CPU only.
Needs third_party/CTformer and third_party/CoreDiff (bash scripts/get_*.sh) plus
timm and einops."""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.data import PatchPairs, SlicePairs, augment, collate_patches, list_pairs, load_input, neighbours
from ldct.models import build_model

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


# ---- fake data: 2 patients, slices whose value is the slice index ----
root = tempfile.mkdtemp()
for p, n in (("L067", 5), ("L096", 3)):
    for kind in ("quarter", "full"):
        os.makedirs(os.path.join(root, p, kind))
        for i in range(n):
            val = i if kind == "quarter" else 100 + i
            np.save(os.path.join(root, p, kind, f"{p}_{i:04d}.npy"), np.full((32, 32), val, np.int16))
pairs = list_pairs(root, ["L067", "L096"])

# ---- neighbours(): clamped at every patient's ends ----
nb = neighbours(pairs, 1)
idx = lambda path: int(os.path.basename(path)[5:9])
check("context 0 = just the slice", neighbours(pairs, 0) == [[p[0]] for p in pairs])
check("context 1, middle slice: i-1, i, i+1", [idx(f) for f in nb[2]] == [1, 2, 3])
check("context 1, first slice of a patient: clamped", [idx(f) for f in nb[0]] == [0, 0, 1])
check("context 1, last slice of a patient: clamped", [idx(f) for f in nb[4]] == [3, 4, 4])
check("context 1 never crosses patients", [idx(f) for f in nb[5]] == [0, 0, 1]
      and all("L096" in f for f in nb[5]))
check("context 2 has 5 channels", len(neighbours(pairs, 2)[2]) == 5)
check("load_input stacks channels", load_input(nb[2]).shape == (3, 32, 32))

# ---- datasets with context ----
ds = PatchPairs(pairs, patch=16, patch_n=2, context=1)
x, y = ds[2]
check("PatchPairs context: x (patch_n, 3, P, P), y (patch_n, 1, P, P)", x.shape == (2, 3, 16, 16) and y.shape == (2, 1, 16, 16))
check("PatchPairs context: centre channel is the slice itself", torch.allclose(x[:, 1], x[:, 1] * 0 + x[0, 1, 0, 0])
      and abs(float(x[0, 1, 0, 0]) - float(torch.tensor((2 + 1024) / 4096.0))) < 1e-6)
xb, yb = collate_patches([ds[0], ds[1]])
check("collate keeps channels", xb.shape == (4, 3, 16, 16) and yb.shape == (4, 1, 16, 16))
lo, hi, patient, name = SlicePairs(pairs, context=1)[0]
check("SlicePairs context: (3,H,W) and (1,H,W)", lo.shape == (3, 32, 32) and hi.shape == (1, 32, 32))
lo0, _, _, _ = SlicePairs(pairs)[0]
check("SlicePairs default unchanged: (1,H,W)", lo0.shape == (1, 32, 32))

# ---- augmentation ----
a = np.arange(2 * 4 * 4).reshape(2, 4, 4)
check("augment mode 0 = identity", np.array_equal(augment(a, 0), a))
check("augment: 8 modes give 8 different images", len({augment(a, m).tobytes() for m in range(8)}) == 8)
check("augment keeps the channel axis", all(augment(a, m).shape == (2, 4, 4) for m in range(8)))
rng_pairs = list_pairs(root, ["L067"])
ds_aug = PatchPairs(rng_pairs, patch=16, patch_n=4, augment=True)
x, y = ds_aug[1]
check("augment: noisy and clean patches still match (same transform)", torch.allclose(y - x, torch.full_like(x, 100 / 4096.0)))

# ---- CTformer wrapper ----
m = build_model("ctformer")
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
check("CTformer: 1,447,477 trainable parameters (paper: 1.45 M)", trainable == 1_447_477)
x = torch.rand(3, 1, 64, 64)
out = m(x)
check("CTformer: (B,1,64,64) in and out", out.shape == x.shape)
loss = m.compute_loss(out, x, x)
check("CTformer loss = MSE*100 + 1e-4", abs(float(loss) - (float(((out - x) ** 2).mean()) * 100 + 1e-4)) < 1e-5)
m.eval()
with torch.no_grad():
    check("CTformer eval is deterministic", torch.allclose(m(x), m(x)))

# ---- CoreDiff wrapper (small image size for speed) ----
m = build_model("corediff", image_size=32)
check("CoreDiff: EMA listed as an uncounted module", m.loss_modules == ("ema",))
check("CoreDiff: EMA parameters frozen", all(not p.requires_grad for p in m.ema.parameters()))
x = torch.rand(2, 3, 32, 32)
t = torch.rand(2, 1, 32, 32)
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 1e-3)
before = [p.detach().clone() for p in m.ema.parameters()]
for _ in range(10):
    loss = m.training_step(x, t)
    opt.zero_grad(); loss.backward(); opt.step()
check("CoreDiff: step counter in the state (resumable)", int(m.n_iter) == 10 and "n_iter" in m.state_dict())
check("CoreDiff: EMA untouched before 10 completed steps", all(torch.equal(a, b) for a, b in zip(before, m.ema.parameters())))
m.training_step(x, t)   # step 11 starts: EMA owed by step 10 -> reset to the model (n < 2000)
check("CoreDiff: EMA reset to the model at step 10 (before start_ema_step)",
      all(torch.equal(a, b) for a, b in zip(m.net.parameters(), m.ema.parameters())))
m2 = build_model("corediff", image_size=32, start_ema_step=0)
m2.n_iter.fill_(10)
w_net = next(m2.net.parameters()); w_net.data.fill_(1.0)
w_ema = next(m2.ema.parameters()); w_ema.data.fill_(0.0)
m2._ema_step(10)
check("CoreDiff: EMA average uses decay 0.995", torch.allclose(w_ema, torch.full_like(w_ema, 0.005)))
m.eval()
out = m.denoise(torch.rand(1, 3, 32, 32))
check("CoreDiff: denoise (1,3,H,W) -> (1,1,H,W) in [0,1]", out.shape == (1, 1, 32, 32) and 0 <= float(out.min()) and float(out.max()) <= 1)
check("CoreDiff: forward == denoise", torch.allclose(m(torch.ones(1, 3, 32, 32)), m.denoise(torch.ones(1, 3, 32, 32))))

print("\nFAILED:" if failed else "\nAll baseline checks passed.", failed or "")
sys.exit(1 if failed else 0)
