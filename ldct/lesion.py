"""Idea 1: make the denoiser care more about tumours.

Two independent pieces, both OFF unless the config turns them on and masks exist:

1. LesionWeightedL1  (train.lesion_weight, default 5.0 when enabled)
   L1 loss in which pixels inside a tumour mask count `weight` times as much:
       loss = sum(w * |pred - target|) / sum(w),   w = weight inside tumours, 1 elsewhere
   Without a mask (or an all-zero mask) it is exactly the plain mean L1, so patients
   without masks (Mayo) and patients with masks (LiTS) can be mixed in one run.
   Models with their own compute_loss (CT-Mamba's Deep NPS loss) keep it and ADD the
   weighted L1 as an extra term scaled by `lesion_extra_scale` (see train.py).

2. Tumour-aware patch sampling  (train.tumour_patch_share, default 0.5 when enabled)
   For slices that have a mask with tumour pixels, that share of the patches is centred
   on a random tumour pixel (shifted so the patch stays inside the image); the rest are
   random as before. Slices without tumours are sampled as before.

Masks live next to the slices: <root>/<patient>/mask/<name>.npy (uint8, 2 = tumour),
as written by scripts/make_lits_lowdose.py. Patients without a mask folder simply
have no masks.
"""
import os

import numpy as np
import torch
import torch.nn as nn

TUMOUR = 2


def mask_path(low_path):
    """<root>/<p>/quarter/<name>.npy -> <root>/<p>/mask/<name>.npy, or None if absent."""
    folder, name = os.path.split(low_path)
    path = os.path.join(os.path.dirname(folder), "mask", name)
    return path if os.path.exists(path) else None


def tumour_mask(mask):
    """Boolean tumour pixels from a label mask (None -> None)."""
    return None if mask is None else np.asarray(mask) == TUMOUR


def weighted_l1(pred, target, tumour, weight):
    """L1 with `weight` on tumour pixels; tumour is a bool/float tensor like pred or None."""
    err = (pred - target).abs()
    if tumour is None or weight == 1.0:
        return err.mean()
    w = 1.0 + (weight - 1.0) * tumour.to(err.dtype)
    return (w * err).sum() / w.sum()


class LesionWeightedL1(nn.Module):
    def __init__(self, weight=5.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, pred, target, tumour=None):
        return weighted_l1(pred, target, tumour, self.weight)


def tumour_patch_corner(tumour, patch, rng=np.random):
    """Top-left corner of a `patch`-sized square centred on a random tumour pixel,
    clipped so the square stays inside the image. None if there is no tumour."""
    ys, xs = np.nonzero(tumour)
    if len(ys) == 0:
        return None
    k = rng.randint(0, len(ys))
    h, w = tumour.shape
    r = int(np.clip(ys[k] - patch // 2, 0, h - patch))
    c = int(np.clip(xs[k] - patch // 2, 0, w - patch))
    return r, c
