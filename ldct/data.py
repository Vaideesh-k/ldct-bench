"""Data loading: reads CT slices from disk and prepares them for the model.

Every slice is saved as an .npy file in Hounsfield units (HU), int16 or float32:
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


def neighbours(pairs, context):
    """For every pair, the noisy slices around it: [i-context, ..., i, ..., i+context]
    of the same patient, as file paths. At a patient's first/last slice the missing
    neighbour is replaced by the edge slice itself ("clamped"), so every slice still
    gets an input. Used by models that take adjacent slices as extra channels (CoreDiff).
    context=0 gives [[low_path]] for every pair."""
    by_patient = {}
    for idx, (_, _, patient, _) in enumerate(pairs):
        by_patient.setdefault(patient, []).append(idx)
    out = [None] * len(pairs)
    for idxs in by_patient.values():
        n = len(idxs)
        for pos, idx in enumerate(idxs):
            out[idx] = [pairs[idxs[min(max(pos + d, 0), n - 1)]][0] for d in range(-context, context + 1)]
    return out


def load_input(low_paths):
    """Stack the noisy slices of one input as channels: (len(low_paths), H, W) in [0, 1]."""
    return hu_to_unit(np.stack([np.load(f).astype(np.float32) for f in low_paths]))


def augment(x, mode):
    """One of the 8 flips/rotations of a (C, H, W) array (mode 0 = unchanged), the same
    set as CTformer's data_aug.py. Applied to noisy and clean patch alike."""
    if mode == 0:
        return x
    x = np.rot90(x, k=mode // 2, axes=(-2, -1))
    return x[..., ::-1, :] if mode % 2 else x


def choose_test_sets(cfg, names=None):
    """Which test sets to run. Named sets must exist in the config. With no
    names, every set whose folder exists is used; missing ones are skipped
    with a warning (e.g. LiTS before it has been prepared)."""
    from ldct.config import resolve
    all_sets = cfg["data"]["test_sets"]
    if names:
        unknown = [n for n in names if n not in all_sets]
        if unknown:
            raise SystemExit(f"Unknown test set(s) {unknown}. Known: {list(all_sets)}")
        return list(names)
    chosen = []
    for name, spec in all_sets.items():
        if os.path.isdir(resolve(spec["root"])):
            chosen.append(name)
        else:
            print(f"WARNING: skipping test set {name}: {spec['root']} does not exist yet")
    if not chosen:
        raise SystemExit("None of the test set folders exist. Prepare the data first.")
    return chosen


def test_set_pairs(test_set):
    """All slice pairs of one test set from the config, e.g.
    {root: data/mayo, patients: [L506]} or {root: data/lits, patients: all}.
    patients: all = every patient folder found under root."""
    from ldct.config import resolve   # imported here to keep data.py usable on its own
    root = str(resolve(test_set["root"]))
    patients = test_set["patients"]
    if isinstance(patients, dict):          # {split: configs/lits_split.json, part: test, prefix: lits_}
        import json
        with open(resolve(patients["split"])) as f:
            split = json.load(f)
        prefix = patients.get("prefix", "lits_")
        patients = [f"{prefix}{k:03d}" for k in split[patients.get("part", "test")]]
    if patients == "all":
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Test set folder {root} does not exist")
        patients = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if not patients:
            raise FileNotFoundError(f"No patient folders in {root}")
    return list_pairs(root, patients)


class PatchPairs(Dataset):
    """Training data: random matching patches from noisy/clean slice pairs.

    context=k  gives the noisy input 2k+1 channels: the k slices above and below
               (clamped at the patient's ends), the target stays the centre slice.
    augment    flips/rotates the second half of each slice's patches at random
               (CTformer's recipe: half the patches as cut, half transformed).
    """

    def __init__(self, pairs, patch=64, patch_n=4, context=0, augment=False,
                 with_masks=False, tumour_share=0.0):
        self.pairs = pairs        # list from list_pairs()
        self.patch = patch        # patch size in pixels (64 -> 64x64)
        self.patch_n = patch_n    # patches cut from each slice
        self.context = context
        self.inputs = neighbours(pairs, context)
        self.augment = augment
        # Idea 1: tumour masks (<patient>/mask/<name>.npy, 2 = tumour) next to the slices
        self.with_masks = with_masks or tumour_share > 0
        self.tumour_share = tumour_share
        if self.with_masks:
            from ldct.lesion import mask_path
            self.masks = [mask_path(pr[0]) for pr in pairs]

    def __len__(self):
        return len(self.pairs)    # how many slices we have

    def __getitem__(self, i):
        # open the slices without loading the whole files into RAM
        los = [np.load(f, mmap_mode="r") for f in self.inputs[i]]
        hi = np.load(self.pairs[i][1], mmap_mode="r")
        h, w = hi.shape
        P = self.patch

        tumour = None
        if self.with_masks and self.masks[i] is not None:
            from ldct.lesion import tumour_mask
            tumour = tumour_mask(np.load(self.masks[i]))
        n_tumour = int(round(self.tumour_share * self.patch_n)) if tumour is not None and tumour.any() else 0

        xs, ys, ms = [], [], []
        for n in range(self.patch_n):
            corner = None
            if n < n_tumour:                      # Idea 1: this patch is centred on a tumour pixel
                from ldct.lesion import tumour_patch_corner
                corner = tumour_patch_corner(tumour, P)
            if corner is None:                    # pick one random top-left corner...
                corner = (np.random.randint(0, h - P + 1), np.random.randint(0, w - P + 1))
            r, c = corner
            # ...and cut the SAME square from every image
            x = np.stack([np.array(lo[r:r + P, c:c + P]) for lo in los])   # (channels, P, P)
            y = np.array(hi[r:r + P, c:c + P])[None]                         # (1, P, P)
            m = (tumour[r:r + P, c:c + P] if tumour is not None else np.zeros((P, P), bool))[None]
            if self.augment and n >= self.patch_n // 2:
                mode = np.random.randint(1, 8)
                x, y, m = augment(x, mode), augment(y, mode), augment(m, mode)
            xs.append(x)
            ys.append(y)
            ms.append(m)

        # stack into (patch_n, channels, P, P), scale to 0-1, convert to PyTorch tensors
        x = torch.from_numpy(np.ascontiguousarray(hu_to_unit(np.stack(xs)).astype(np.float32)))
        y = torch.from_numpy(np.ascontiguousarray(hu_to_unit(np.stack(ys)).astype(np.float32)))
        if self.with_masks:
            return x, y, torch.from_numpy(np.ascontiguousarray(np.stack(ms)))   # (patch_n, 1, P, P) bool
        return x, y


def collate_patches(batch):
    """Stack 8 items of (4,C,64,64) into one batch of (32,C,64,64); masks too if present."""
    x = torch.cat([b[0] for b in batch])   # all noisy patches, one after another
    y = torch.cat([b[1] for b in batch])   # all clean patches, in the same order
    if len(batch[0]) == 3:
        return x, y, torch.cat([b[2] for b in batch])
    return x, y


class SlicePairs(Dataset):
    """Validation/test data: whole noisy and clean slices, plus their names.
    context=k gives the noisy input 2k+1 channels (see PatchPairs)."""

    def __init__(self, pairs, context=0):
        self.pairs = pairs    # list from list_pairs()
        self.inputs = neighbours(pairs, context)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        _, hi_path, patient, name = self.pairs[i]
        lo = load_input(self.inputs[i])                                  # (channels, H, W)
        hi = hu_to_unit(np.load(hi_path).astype(np.float32))[None]       # (1, H, W)
        return torch.from_numpy(lo), torch.from_numpy(hi), patient, name
