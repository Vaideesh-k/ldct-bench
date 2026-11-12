"""Checks for Idea 1 (ldct/lesion.py + the hooks in ldct/data.py and scripts/train.py)."""
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ldct.data import PatchPairs, collate_patches, list_pairs, test_set_pairs
from ldct.lesion import LesionWeightedL1, mask_path, tumour_mask, tumour_patch_corner, weighted_l1
from train import extra_patients, training_loss

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


# ---- the loss ----
pred = torch.zeros(2, 1, 8, 8)
target = torch.ones(2, 1, 8, 8)
tum = torch.zeros(2, 1, 8, 8, dtype=torch.bool); tum[0, 0, :4, :4] = True     # 16 of 128 pixels
check("no mask -> plain mean L1", float(weighted_l1(pred, target, None, 5.0)) == 1.0)
check("all-zero mask -> plain mean L1", float(weighted_l1(pred, target, torch.zeros_like(tum), 5.0)) == 1.0)
check("constant error -> weighting changes nothing", abs(float(weighted_l1(pred, target, tum, 5.0)) - 1.0) < 1e-6)
pred2 = target.clone(); pred2[tum] = 0.0                                       # error only inside tumours
plain = float(F.l1_loss(pred2, target))
w = float(LesionWeightedL1(5.0)(pred2, target, tum))
check(f"error only inside tumours counts 5x: plain {plain:.4f} -> weighted {w:.4f}", abs(w - (5 * 16) / (5 * 16 + 112)) < 1e-6 and w > plain)
pred3 = target.clone(); pred3[~tum] = 0.0                                      # error only outside
check("error only outside tumours counts less", float(LesionWeightedL1(5.0)(pred3, target, tum)) < float(F.l1_loss(pred3, target)))
check("weight 1 == plain L1", abs(float(weighted_l1(pred2, target, tum, 1.0)) - plain) < 1e-7)

# ---- sampling helpers ----
t = np.zeros((64, 64), bool); t[50:54, 3:6] = True
rng = np.random.RandomState(0)
corners = [tumour_patch_corner(t, 16, rng) for _ in range(50)]
check("tumour patch corners stay inside the image", all(0 <= r <= 48 and 0 <= c <= 48 for r, c in corners))
check("tumour patch always contains tumour pixels", all(t[r:r + 16, c:c + 16].any() for r, c in corners))
check("no tumour -> None", tumour_patch_corner(np.zeros((8, 8), bool), 4) is None)
check("tumour_mask: label 2 only", tumour_mask(np.array([[0, 1, 2]])).tolist() == [[False, False, True]] and tumour_mask(None) is None)

# ---- dataset with masks: fake patient with a tumour in one corner ----
root = tempfile.mkdtemp()
for kind in ("quarter", "full", "mask"):
    os.makedirs(os.path.join(root, "lits_000", kind))
for i in range(2):
    np.save(os.path.join(root, "lits_000", "quarter", f"lits_000_{i:04d}.npy"), np.zeros((128, 128), np.int16))
    np.save(os.path.join(root, "lits_000", "full", f"lits_000_{i:04d}.npy"), np.zeros((128, 128), np.int16))
    m = np.zeros((128, 128), np.uint8); m[100:110, 100:110] = 2 if i == 0 else 1
    np.save(os.path.join(root, "lits_000", "mask", f"lits_000_{i:04d}.npy"), m)
pairs = list_pairs(root, ["lits_000"])
check("mask_path finds the mask next to the slice", mask_path(pairs[0][0]).endswith("mask/lits_000_0000.npy"))
check("mask_path: None when absent", mask_path("/nowhere/p/quarter/x.npy") is None)
np.random.seed(0)
ds = PatchPairs(pairs, patch=32, patch_n=4, tumour_share=0.5)
x, y, m = ds[0]
check("with masks: 3 outputs, mask (patch_n,1,P,P) bool", x.shape == (4, 1, 32, 32) and m.shape == (4, 1, 32, 32) and m.dtype == torch.bool)
check("tumour_share 0.5: the first 2 patches contain tumour", bool(m[0].any()) and bool(m[1].any()))
x, y, m = ds[1]                                                              # slice 1: liver only, no tumour
check("slice without tumour: masks all False, random patches", not m.any())
xb, yb, mb = collate_patches([ds[0], ds[1]])
check("collate stacks masks too", mb.shape == (8, 1, 32, 32))
ds0 = PatchPairs(pairs, patch=32, patch_n=4)
check("default dataset unchanged: 2 outputs", len(ds0[0]) == 2)
ds_aug = PatchPairs(pairs, patch=32, patch_n=4, augment=True, tumour_share=1.0)
x, y, m = ds_aug[0]
check("augmentation keeps every tumour patch's mask non-empty", all(bool(mm.any()) for mm in m))

# ---- training_loss hook ----
class L1Model(torch.nn.Module):
    def forward(self, x):
        return x * 0.0
class OwnLoss(L1Model):
    def compute_loss(self, pred, target, inp):
        return F.mse_loss(pred, target)
inp = torch.rand(2, 1, 8, 8); tgt = torch.ones(2, 1, 8, 8); tgt[tum] = 0.0
loss_fn = LesionWeightedL1(5.0)
check("plain model + lesion loss = weighted L1", torch.allclose(training_loss(L1Model(), inp, tgt, tum, loss_fn), loss_fn(torch.zeros_like(tgt), tgt, tum)))
check("plain model without lesion loss = L1", torch.allclose(training_loss(L1Model(), inp, tgt), F.l1_loss(torch.zeros_like(tgt), tgt)))
own = training_loss(OwnLoss(), inp, tgt, tum, loss_fn, extra_scale=100.0)
check("own-loss model: loss + 100 x weighted L1", torch.allclose(own, F.mse_loss(torch.zeros_like(tgt), tgt) + 100 * loss_fn(torch.zeros_like(tgt), tgt, tum)))

# ---- split handling: extra train sets and the locked LiTS test set ----
split = {"train": [0, 1, 3], "test": [2]}
sp = os.path.join(root, "split.json"); json.dump(split, open(sp, "w"))
check("extra_patients from split: train part only", extra_patients({"root": root, "split": sp, "part": "train"}) == ["lits_000", "lits_001", "lits_003"])
try:
    extra_patients({"root": root, "split": sp, "part": "test"}); check("extra_patients refuses the test part", False)
except SystemExit:
    check("extra_patients refuses the test part", True)
os.makedirs(os.path.join(root, "lits_002", "quarter")); os.makedirs(os.path.join(root, "lits_002", "full"))
np.save(os.path.join(root, "lits_002", "quarter", "lits_002_0000.npy"), np.zeros((8, 8), np.int16))
np.save(os.path.join(root, "lits_002", "full", "lits_002_0000.npy"), np.zeros((8, 8), np.int16))
tp = test_set_pairs({"root": root, "patients": {"split": sp, "part": "test", "prefix": "lits_"}})
check("test set from split uses only the locked test patients", [p[2] for p in tp] == ["lits_002"])

print("\nFAILED:" if failed else "\nAll Idea-1 checks passed.", failed or "")
sys.exit(1 if failed else 0)
