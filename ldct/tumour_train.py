"""Shared training pieces for the REFINER and its CONTROL (final-week steps 3 and 4).

Both runs use exactly the same data, sampling and loss; only the model differs.

DATA (MixedPatches): an endless, deterministic stream of patch triples. Patch i is a pure
function of (seed, i), so a resumed run continues the same stream.
  - even i: a random 9-patient Mayo training slice, random position
  - odd i:  a random LiTS TRAINING patient slice (locked test patients are refused);
            with probability `tumour_share` (0.3) the patch is centred on a random tumour
            pixel of a random tumour slice, otherwise a random slice and position
  A batch of 16 consecutive indices is therefore 8 Mayo + 8 LiTS patches.
  Inputs: [quarter] (control) or [quarter, Stage A] (refiner), in [0, 1] via hu_to_unit.
  Stage A = the frozen CT-Mamba output, precomputed next to the inputs as
  <root>/<patient>/ct_mamba/<name>.npy (int16 HU, scripts/precompute_stage_a.py).

LOSS (l1_lesion): L1 over the whole patch + (weight - 1) x L1 inside the tumour mask, both
divided by the number of pixels, so a tumour pixel counts `weight` (5) times in total and a
patch without tumour gets the plain L1. Masks are the LiTS tumour labels dilated by 3
pixels (square 7x7 neighbourhood) to include the edges; Mayo has no masks.
Masks are used ONLY here, in the training loss. Inference never sees them.
"""
import json
import os

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from ldct.data import hu_to_unit, list_pairs
from ldct.lesion import mask_path

STAGE_A = "ct_mamba"            # folder name of the precomputed Stage-A slices


def stage_a_path(low_path, folder=STAGE_A):
    d, name = os.path.split(low_path)
    return os.path.join(os.path.dirname(d), folder, name)


def dilated_tumour(mask, pixels=3):
    """Tumour (label 2) dilated by `pixels` in every direction (7x7 square for 3)."""
    t = np.asarray(mask) == 2
    return binary_dilation(t, np.ones((3, 3), bool), iterations=pixels) if pixels and t.any() else t


def l1_lesion(pred, target, mask=None, weight=5.0):
    """mean|e| + (weight-1) * mean(mask * |e|): tumour pixels count `weight` times."""
    err = (pred - target).abs()
    loss = err.mean()
    if mask is not None and weight != 1.0:
        loss = loss + (weight - 1.0) * (err * mask.to(err.dtype)).mean()
    return loss


def lits_train_patients(split_file, prefix="lits_"):
    split = json.load(open(split_file))
    train, test = set(split["train"]), set(split["test"])
    assert not train & test, "split file: train and test overlap"
    return [f"{prefix}{k:03d}" for k in sorted(train)]


class MixedPatches(Dataset):
    def __init__(self, mayo_root, mayo_patients, lits_root, lits_patients, patch=128,
                 n_items=10 ** 9, tumour_share=0.3, stage_a=False, dilate=3, seed=0,
                 locked=()):
        bad = set(lits_patients) & set(locked)
        if bad or "L506" in mayo_patients:
            raise ValueError(f"locked test patients in the training data: {sorted(bad) or 'L506'}")
        self.mayo = list_pairs(mayo_root, mayo_patients)
        self.lits = list_pairs(lits_root, lits_patients)
        self.patch, self.n_items, self.share = patch, n_items, tumour_share
        self.stage_a, self.dilate, self.seed = stage_a, dilate, seed
        self.lits_masks = [mask_path(p[0]) for p in self.lits]
        # slices with tumour: found once from the masks (uint8, 2 = tumour)
        self.tumour_slices = [j for j, m in enumerate(self.lits_masks)
                              if m is not None and (np.load(m, mmap_mode="r") == 2).any()]
        if tumour_share > 0 and not self.tumour_slices:
            raise ValueError("tumour_share > 0 but no LiTS training slice has a tumour")
        if stage_a:
            missing = [stage_a_path(p[0]) for p in self.mayo + self.lits if not os.path.exists(stage_a_path(p[0]))]
            if missing:
                raise FileNotFoundError(f"{len(missing)} Stage-A slices missing, e.g. {missing[0]} "
                                        f"(run scripts/precompute_stage_a.py)")

    def __len__(self):
        return self.n_items

    def _cut(self, pair, mask_file, rng, centre_on_tumour):
        low = np.load(pair[0], mmap_mode="r")
        h, w = low.shape
        P = self.patch
        mask = np.load(mask_file) if mask_file is not None else None
        if centre_on_tumour:
            ys, xs = np.nonzero(mask == 2)
            k = rng.integers(len(ys))
            r = int(np.clip(ys[k] - P // 2, 0, h - P))
            c = int(np.clip(xs[k] - P // 2, 0, w - P))
        else:
            r, c = int(rng.integers(0, h - P + 1)), int(rng.integers(0, w - P + 1))
        sl = (slice(r, r + P), slice(c, c + P))
        chans = [np.asarray(low[sl], np.float32)]
        if self.stage_a:
            chans.append(np.asarray(np.load(stage_a_path(pair[0]), mmap_mode="r")[sl], np.float32))
        x = hu_to_unit(np.stack(chans))
        y = hu_to_unit(np.asarray(np.load(pair[1], mmap_mode="r")[sl], np.float32))[None]
        m = (dilated_tumour(mask, self.dilate)[sl] if mask is not None
             else np.zeros((P, P), bool))[None]
        return x, y, m

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, int(i)))
        if i % 2 == 0:                                   # Mayo
            pair, mfile, centre = self.mayo[rng.integers(len(self.mayo))], None, False
        elif rng.random() < self.share:                  # LiTS, centred on a tumour
            j = self.tumour_slices[rng.integers(len(self.tumour_slices))]
            pair, mfile, centre = self.lits[j], self.lits_masks[j], True
        else:                                            # LiTS, anywhere
            j = int(rng.integers(len(self.lits)))
            pair, mfile, centre = self.lits[j], self.lits_masks[j], False
        x, y, m = self._cut(pair, mfile, rng, centre)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)), int(i % 2))


class FromIndex(torch.utils.data.Sampler):
    """Indices start, start+1, ... (resume = start after the last trained item)."""

    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self):
        return self.stop - self.start
