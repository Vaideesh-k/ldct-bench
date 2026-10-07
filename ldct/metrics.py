"""Image quality scores: how close a denoised slice is to the full-dose slice.

All scores are computed in a soft-tissue window (-160..240 HU by default),
the same window the CT-Mamba paper reports, so every model is judged alike.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def score(pred_hu, clean_hu, lo=-160, hi=240):
    """Compare a denoised slice with the clean slice, both in HU.

    Both images are clipped to [lo, hi] first, so values outside the
    window (air, bone) do not affect the scores.
    Returns {"psnr": dB (higher is better),
             "ssim": 0..1 (higher is better),
             "rmse": HU (lower is better)}.
    """
    pred = np.clip(np.asarray(pred_hu, dtype=np.float64), lo, hi)
    clean = np.clip(np.asarray(clean_hu, dtype=np.float64), lo, hi)
    if pred.shape != clean.shape:
        raise ValueError(f"Shapes differ: prediction {pred.shape} vs clean {clean.shape}")

    data_range = hi - lo   # the largest possible difference inside the window
    with np.errstate(divide="ignore"):   # identical images: PSNR is infinite, not an error
        psnr = peak_signal_noise_ratio(clean, pred, data_range=data_range)
    return {
        "psnr": float(psnr),
        "ssim": float(structural_similarity(clean, pred, data_range=data_range)),
        "rmse": float(np.sqrt(np.mean((pred - clean) ** 2))),
    }
