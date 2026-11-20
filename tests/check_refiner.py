"""Checks for the refiner + control (ldct/models/ct_mamba_refiner.py, ldct/tumour_train.py,
scripts/train_tumour.py). CPU only; CT-Mamba itself is not needed."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ldct.data import hu_to_unit
from ldct.models import build_model
from ldct.models.ct_mamba_refiner import Refiner
from ldct.tumour_train import MixedPatches, dilated_tumour, l1_lesion, stage_a_path

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


torch.manual_seed(0)
# ---- model ----
net = build_model("ct_mamba_refiner")
n_params = sum(p.numel() for p in net.parameters())
print(f"refiner parameters: {n_params:,}")
check("parameter count in the expected range for a 32/64/128/256 U-Net (1.5-2.5 M)", 1.5e6 < n_params < 2.5e6)
x = torch.rand(3, 2, 128, 128)
y = net(x)
check("output shape (B, 1, H, W) for 128 patches", tuple(y.shape) == (3, 1, 128, 128))
check("output shape for a whole 512 slice", tuple(net(torch.rand(1, 2, 512, 512)).shape) == (1, 1, 512, 512))
check("zero-init: untrained refiner == Stage A EXACTLY", torch.equal(y, x[:, 1:2]))
check("zero-init: final conv weights and bias are 0", float((net.head.weight.abs().sum() + net.head.bias.abs().sum()).detach()) == 0.0)
try:
    net(torch.rand(1, 1, 64, 64)); check("1-channel input refused", False)
except ValueError:
    check("1-channel input refused", True)
with torch.no_grad():
    net.head.bias.fill_(0.01)
check("residual is ADDED to Stage A", torch.allclose(net(x), x[:, 1:2] + net.residual(x)) and
      torch.allclose(net(x) - x[:, 1:2], torch.full_like(x[:, 1:2], 0.01) + 0 * x[:, 1:2], atol=1e-6))
with torch.no_grad():
    net.head.bias.zero_()
# gradients reach the U-Net even though the head starts at zero
loss = (net(x) - torch.rand(3, 1, 128, 128)).abs().mean(); loss.backward()
check("zero-init head still gets a gradient (the model can learn)", float(net.head.weight.grad.abs().sum()) > 0)
with torch.no_grad():
    torch.nn.init.normal_(net.head.weight, std=0.1)
    x2 = x.clone(); x2[:, 0] += 0.1
    check("noisy input changes the residual (channel 0 is used)", not torch.allclose(net(x), net(x2)))
    net.head.weight.zero_()

# ---- loss ----
pred, target = torch.zeros(2, 1, 8, 8), torch.ones(2, 1, 8, 8)
m = torch.zeros(2, 1, 8, 8, dtype=torch.bool); m[0, 0, :4, :4] = True          # 16 of 128 pixels
check("no mask -> plain L1", float(l1_lesion(pred, target, None)) == 1.0)
check("all-zero mask -> plain L1", float(l1_lesion(pred, target, torch.zeros_like(m))) == 1.0)
err_in = torch.zeros(2, 1, 8, 8); err_in[m] = 1.0
check("error only inside mask: 16 px x 5 / 128 px", abs(float(l1_lesion(err_in, torch.zeros_like(err_in), m)) - 16 * 5 / 128) < 1e-6)
err_out = torch.ones(2, 1, 8, 8); err_out[m] = 0.0
check("error only outside mask: weighting NOT applied there (= plain L1)",
      abs(float(l1_lesion(err_out, torch.zeros_like(err_out), m)) - float(err_out.mean())) < 1e-7)
check("weight 1 -> plain L1", float(l1_lesion(err_in, torch.zeros_like(err_in), m, 1.0)) == float(err_in.mean()))

# ---- mask dilation ----
lab = np.zeros((20, 20), np.uint8); lab[10, 10] = 2; lab[2, 2] = 1
d = dilated_tumour(lab, 3)
check("dilation by 3 px = 7x7 square around a tumour pixel", d.sum() == 49 and d[7:14, 7:14].all())
check("liver (label 1) is not tumour", not d[2, 2])
check("no tumour -> empty mask", not dilated_tumour(np.ones((5, 5), np.uint8), 3).any())

# ---- data ----
tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
def write(root, p, n, tumour=False):
    for kind in ("quarter", "full", "ct_mamba") + (("mask",) if root.endswith("lits") else ()):
        os.makedirs(os.path.join(root, p, kind), exist_ok=True)
    for i in range(n):
        name = f"{p}_{i:03d}.npy"
        full = rng.normal(40, 10, (160, 160)).astype(np.int16)
        np.save(os.path.join(root, p, "full", name), full)
        np.save(os.path.join(root, p, "quarter", name), (full + rng.normal(0, 50, full.shape)).astype(np.int16))
        np.save(os.path.join(root, p, "ct_mamba", name), (full + 1).astype(np.int16))
        if root.endswith("lits"):
            msk = np.zeros((160, 160), np.uint8); msk[40:120, 40:120] = 1
            if tumour and i % 2 == 0:
                msk[100:104, 30:34] = 2
            np.save(os.path.join(root, p, "mask", name), msk)
mayo, lits = os.path.join(tmp, "mayo"), os.path.join(tmp, "lits")
write(mayo, "L067", 4); write(lits, "lits_000", 6, tumour=True); write(lits, "lits_001", 4)
ds = MixedPatches(mayo, ["L067"], lits, ["lits_000", "lits_001"], patch=64, n_items=4000,
                  tumour_share=0.3, stage_a=True, seed=0, locked=["lits_002"])
x, y, mk, src = ds[1]
check("refiner item: input (2, P, P), target (1, P, P), mask (1, P, P)",
      tuple(x.shape) == (2, 64, 64) and tuple(y.shape) == (1, 64, 64) and tuple(mk.shape) == (1, 64, 64))
check("inputs are in [0, 1] units (hu_to_unit)", abs(float(x[0].mean()) - float(hu_to_unit(np.float32(40)))) < 0.02)
check("channel 1 is the Stage-A slice (= full + 1 HU in the fake data)", torch.allclose(x[1], y[0] + 1.0 / 4096, atol=1e-6))
check("same index -> same patch (deterministic, resumable)", all(torch.equal(a, b) for a, b in zip(ds[7][:3], ds[7][:3])))
srcs = [ds[i][3] for i in range(16)]
check("a batch of 16 = 8 Mayo + 8 LiTS", srcs.count(0) == 8 and srcs.count(1) == 8)
check("Mayo patches never carry a mask", all(not ds[i][2].any() for i in range(0, 200, 2)))
lits_items = [ds[i] for i in range(1, 4000, 2)]
share = np.mean([float(it[2].any()) for it in lits_items])
check(f"about 30% of LiTS patches contain tumour (got {share:.2f})", 0.25 <= share <= 0.45)
check("masks only where tumour (patches from tumour-free patient have empty masks)",
      not any(ds._cut(ds.lits[j], ds.lits_in[j], ds.lits_masks[j], np.random.default_rng(j), False)[2].any()
              for j in range(len(ds.lits)) if "lits_001" in ds.lits[j][0]))
try:
    MixedPatches(mayo, ["L067"], lits, ["lits_000"], locked=["lits_000"]); check("locked LiTS test patient refused", False)
except ValueError:
    check("locked LiTS test patient refused", True)
try:
    MixedPatches(mayo, ["L506"], lits, ["lits_000"]); check("L506 refused", False)
except ValueError:
    check("L506 refused", True)
check("Stage-A path is <patient>/ct_mamba/<name>", stage_a_path("/a/p/quarter/x.npy") == "/a/p/ct_mamba/x.npy")
os.environ["LDCT_STAGE_A_ROOT"] = "/scratch/s"
check("Stage-A path under $LDCT_STAGE_A_ROOT/<patient>/ct_mamba/<name>", stage_a_path("/a/p/quarter/x.npy") == "/scratch/s/p/ct_mamba/x.npy")
del os.environ["LDCT_STAGE_A_ROOT"]
ctl = MixedPatches(mayo, ["L067"], lits, ["lits_000"], patch=64, n_items=10, stage_a=False)
check("control item: 1 input channel", ctl[1][0].shape[0] == 1)

# ---- 2.5D variant: [z-1, z, z+1, Stage A] ----
ds3 = MixedPatches(mayo, ["L067"], lits, ["lits_000", "lits_001"], patch=64, n_items=4000,
                   tumour_share=0.3, stage_a=True, seed=0, locked=["lits_002"], context=1)
same_patch = all(torch.equal(ds3[i][0][1], ds[i][0][0]) and torch.equal(ds3[i][0][3], ds[i][0][1]) and
                 torch.equal(ds3[i][1], ds[i][1]) and torch.equal(ds3[i][2], ds[i][2]) for i in range(40))
check("2.5D item: 4 input channels", tuple(ds3[1][0].shape) == (4, 64, 64))
check("2.5D: centre slice, Stage A, target and mask identical to the 2D sampler (same patches)", same_patch)
nb = ds3.mayo_in
check("2.5D: neighbours are z-1/z/z+1 of the same patient, clamped at the edges",
      nb[0] == [ds3.mayo[0][0], ds3.mayo[0][0], ds3.mayo[1][0]] and nb[1] == [ds3.mayo[0][0], ds3.mayo[1][0], ds3.mayo[2][0]]
      and nb[-1][2] == ds3.mayo[-1][0])
check("2.5D: LiTS neighbours never cross patients",
      all(all(os.path.dirname(f) == os.path.dirname(pr[0]) for f in lows) for pr, lows in zip(ds3.lits, ds3.lits_in)))
net4 = Refiner(in_channels=4)
x4 = torch.rand(2, 4, 64, 64)
check("2.5D model: untrained output == Stage A (last channel) EXACTLY", torch.equal(net4(x4), x4[:, 3:4]))
try:
    net4(torch.rand(1, 2, 64, 64)); check("2.5D model refuses 2-channel input", False)
except ValueError:
    check("2.5D model refuses 2-channel input", True)

# ---- end-to-end smoke of the training script (refiner, CPU) ----
cfg = os.path.join(tmp, "cfg.yaml")
split = os.path.join(tmp, "split.json")
open(split, "w").write('{"train": [0, 1], "test": [2]}')
open(cfg, "w").write(f"""seed: 0
mayo_root: {mayo}
mayo_patients: [L067]
lits_root: {lits}
lits_split: {split}
iterations: 40
lr: 2.0e-4
lr_min: 1.0e-6
weight_decay: 1.0e-4
betas: [0.9, 0.999]
lesion_weight: 5.0
dilate: 3
tumour_share: 0.3
window: 10
workers: 0
modes:
  refiner: {{model: ct_mamba_refiner, patch: 64, batch: 4, stage_a: true}}
  refiner25d: {{model: ct_mamba_refiner, model_args: {{in_channels: 4}}, context: 1, patch: 64, batch: 4, stage_a: true}}
""")
env = dict(os.environ, PYTHONPATH=ROOT)
r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_tumour.py"), "--mode", "refiner",
                    "--config", cfg, "--run", "check_refiner_tmp", "--smoke", "--device", "cpu"],
                   capture_output=True, text=True, cwd=ROOT, env=env)
print(r.stdout[-600:], r.stderr[-1500:])
run_dir = os.path.join(ROOT, "runs", "check_refiner_tmp_smoke")
check("training script runs and writes best.pth + 4 log rows",
      r.returncode == 0 and os.path.exists(os.path.join(run_dir, "best.pth")) and
      len(open(os.path.join(run_dir, "train_log.csv")).read().strip().splitlines()) == 5)
import shutil
shutil.rmtree(run_dir, ignore_errors=True)
r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_tumour.py"), "--mode", "refiner25d",
                    "--config", cfg, "--run", "check_refiner25d_tmp", "--smoke", "--device", "cpu"],
                   capture_output=True, text=True, cwd=ROOT, env=env)
print(r.stdout[-400:], r.stderr[-1500:])
run_dir = os.path.join(ROOT, "runs", "check_refiner25d_tmp_smoke")
ok = r.returncode == 0 and os.path.exists(os.path.join(run_dir, "best.pth"))
check("2.5D training script runs; checkpoint records in_channels=4",
      ok and torch.load(os.path.join(run_dir, "best.pth"), map_location="cpu")["model_args"] == {"in_channels": 4})
shutil.rmtree(run_dir, ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
