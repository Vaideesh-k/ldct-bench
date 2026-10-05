"""Data loading: reads CT slices from disk and prepares them for the model.

Every slice is saved as a float32 .npy file in Hounsfield units (HU):
    <root>/<patient>/quarter/<name>.npy   low-dose (noisy) slice
    <root>/<patient>/full/<name>.npy      normal-dose (clean) slice, same file name
"""
from glob import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# The range of CT values we scale from. Same as the CT-Mamba preprocessed data.
HU_MIN, HU_MAX = -1024.0, 3072.0


def hu_to_unit(x):
    """Hounsfield units -> 0 to 1 (what the model sees)."""
    return (x - HU_MIN) / (HU_MAX - HU_MIN)


def unit_to_hu(x):
    """0 to 1 (model output) -> Hounsfield units (for saving and scoring)."""
    return x * (HU_MAX - HU_MIN) + HU_MIN

def list_pairs(root, patients, low="quarter", high="full", every=1):
    """Find matching noisy/clean slice files.

    Returns a list of (low_path, high_path, patient, name).
    every=k keeps only every k-th slice (used for a small validation set).
    """
    pairs = []
    for p in patients:
        # all noisy slices for this patient, sorted so the order is always the same
        lows = sorted(glob(os.path.join(root, p, low, "*.npy")))
        if not lows:
            raise FileNotFoundError(f"No '{low}' slices for patient {p} under {root}")

        for f in lows[::every]:
            # the clean partner has the same file name, in the other folder
            g = os.path.join(root, p, high, os.path.basename(f))
            if not os.path.exists(g):
                raise FileNotFoundError(f"Missing clean slice for {f}: expected {g}")
            name = os.path.splitext(os.path.basename(f))[0]   # "L067_0000.npy" -> "L067_0000"
            pairs.append((f, g, p, name))
    return pairs
class PatchPairs(Dataset):
    """Training data: random matching patches from noisy/clean slice pairs."""

    def __init__(self, pairs, patch=64, patch_n=4):
        self.pairs = pairs        # list from list_pairs()
        self.patch = patch        # patch size in pixels (64 -> 64x64)
        self.patch_n = patch_n    # patches cut from each slice

    def __len__(self):
        return len(self.pairs)    # how many slices we have

    def __getitem__(self, i):
        # open slice pair i without loading the whole file into RAM
        lo = np.load(self.pairs[i][0], mmap_mode="r")
        hi = np.load(self.pairs[i][1], mmap_mode="r")
        h, w = lo.shape
        P = self.patch

        xs, ys = [], []
        for _ in range(self.patch_n):
            # pick one random top-left corner...
            r = np.random.randint(0, h - P + 1)
            c = np.random.randint(0, w - P + 1)
            # ...and cut the SAME square from both images
            xs.append(np.array(lo[r:r + P, c:c + P]))
            ys.append(np.array(hi[r:r + P, c:c + P]))

        # stack into (patch_n, 1, P, P), scale to 0-1, convert to PyTorch tensors
        x = torch.from_numpy(hu_to_unit(np.stack(xs)).astype(np.float32))[:, None]
        y = torch.from_numpy(hu_to_unit(np.stack(ys)).astype(np.float32))[:, None]
        return x, y
def collate_patches(batch):
    """Stack 8 items of (4,1,64,64) into one batch of (32,1,64,64)."""
    x = torch.cat([b[0] for b in batch])   # all noisy patches, one after another
    y = torch.cat([b[1] for b in batch])   # all clean patches, in the same order
    return x, y
class SlicePairs(Dataset):
    """Validation/test data: whole noisy and clean slices, plus their names."""

    def __init__(self, pairs):
        self.pairs = pairs    # list from list_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lo_path, hi_path, patient, name = self.pairs[i]
        # load the full slices and scale to 0-1
        lo = hu_to_unit(np.load(lo_path).astype(np.float32))
        hi = hu_to_unit(np.load(hi_path).astype(np.float32))
        # add a channel dimension: (H, W) -> (1, H, W)
        return torch.from_numpy(lo)[None], torch.from_numpy(hi)[None], patient, name