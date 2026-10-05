"""Quick checks for ldct/data.py. Run with the ▶ button at the top right."""
import os
import sys
import tempfile

import numpy as np

# let this file import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.data import hu_to_unit, list_pairs, unit_to_hu


def make_fake_data():
    """2 patients x 4 slices, in the same folder layout as the real data."""
    root = tempfile.mkdtemp()
    for p in ["L067", "L096"]:
        for kind in ["quarter", "full"]:
            os.makedirs(os.path.join(root, p, kind))
            for i in range(4):
                np.save(os.path.join(root, p, kind, f"{p}_{i:04d}.npy"), np.zeros((8, 8), np.float32))
    return root


# Part 1: unit conversion
print("Part 1:", hu_to_unit(-1024), hu_to_unit(0), hu_to_unit(3072), unit_to_hu(hu_to_unit(40.0)))

# Part 2: finding pairs
root = make_fake_data()
pairs = list_pairs(root, ["L067", "L096"])
print("Part 2: all slices:", len(pairs))
print("Part 2: first pair:", pairs[0][2], pairs[0][3])
print("Part 2: every 2nd:", [x[3] for x in list_pairs(root, ["L067"], every=2)])

os.remove(os.path.join(root, "L096", "full", "L096_0002.npy"))
try:
    list_pairs(root, ["L096"])
except FileNotFoundError:
    print("Part 2: caught missing file: OK")
try:
    list_pairs(root, ["L506"])
except FileNotFoundError:
    print("Part 2: caught missing patient: OK")
# Part 3: training patches
from ldct.data import PatchPairs

root3 = tempfile.mkdtemp()
for kind in ["quarter", "full"]:
    os.makedirs(os.path.join(root3, "L067", kind))
img = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)   # every pixel has a unique value
np.save(os.path.join(root3, "L067", "quarter", "L067_0000.npy"), img)
np.save(os.path.join(root3, "L067", "full", "L067_0000.npy"), img + 1000)

ds = PatchPairs(list_pairs(root3, ["L067"]), patch=64, patch_n=4)
x, y = ds[0]
print("Part 3: dataset size:", len(ds))
print("Part 3: shapes:", tuple(x.shape), tuple(y.shape))
same_spot = np.allclose(unit_to_hu(y.numpy()) - unit_to_hu(x.numpy()), 1000, atol=0.1)
print("Part 3: same spot in both images:", same_spot)
# Part 4: batching
from torch.utils.data import DataLoader
from ldct.data import collate_patches

ds8 = PatchPairs(list_pairs(root3, ["L067"]) * 8, patch=64, patch_n=4)   # same slice listed 8 times
loader = DataLoader(ds8, batch_size=8, collate_fn=collate_patches)
xb, yb = next(iter(loader))
print("Part 4: batch shapes:", tuple(xb.shape), tuple(yb.shape))
still_matched = np.allclose(unit_to_hu(yb.numpy()) - unit_to_hu(xb.numpy()), 1000, atol=0.1)
print("Part 4: patches still matched:", still_matched)
# Part 5: full slices for testing
from ldct.data import SlicePairs

full_ds = SlicePairs(list_pairs(root3, ["L067"]))
lo, hi, patient, name = full_ds[0]
print("Part 5: shapes:", tuple(lo.shape), tuple(hi.shape))
print("Part 5: names:", patient, name)
# realistic CT range: air (-1024) up to dense bone (3072)
real = np.linspace(-1024, 3072, 128 * 128, dtype=np.float32).reshape(128, 128)
root5 = tempfile.mkdtemp()
for kind in ["quarter", "full"]:
    os.makedirs(os.path.join(root5, "L067", kind))
    np.save(os.path.join(root5, "L067", kind, "L067_0000.npy"), real)
lo5, hi5, _, _ = SlicePairs(list_pairs(root5, ["L067"]))[0]
print("Part 5: values in 0-1:", float(lo5.min()) >= 0 and float(hi5.max()) <= 1)
print("Part 5: back to HU matches file:", np.allclose(unit_to_hu(lo.numpy()[0]), img, atol=0.1))