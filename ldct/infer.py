"""Inference: denoise one whole slice with a trained model."""
import torch
import torch.nn.functional as F


def _starts(size, patch, stride):
    """Top/left corners along one side: every `stride` pixels, plus one at the
    very end so the last row/column is always covered."""
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def tiled(model, x, patch, stride, batch=64):
    """Denoise a (1,1,H,W) slice patch by patch and average where patches overlap.

    Used for models trained on small patches that cannot take a whole slice
    (e.g. CT-Mamba). Slices smaller than one patch are padded, then cropped back.
    """
    if x.dim() != 4 or x.shape[:2] != (1, 1):
        raise ValueError(f"tiled() expects one slice shaped (1,1,H,W), got {tuple(x.shape)}")
    H, W = x.shape[-2:]

    # pad tiny slices up to one patch (copying the edge pixels)
    pad_h, pad_w = max(0, patch - H), max(0, patch - W)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    h, w = x.shape[-2:]

    corners = [(r, c) for r in _starts(h, patch, stride) for c in _starts(w, patch, stride)]

    # running sum of outputs and how many patches covered each pixel.
    # Kept in float64 on the CPU so averaging identical values is exact.
    total = torch.zeros(h, w, dtype=torch.float64)
    count = torch.zeros(h, w, dtype=torch.float64)

    with torch.no_grad():
        for i in range(0, len(corners), batch):
            group = corners[i:i + batch]
            patches = torch.cat([x[..., r:r + patch, c:c + patch] for r, c in group])   # (n,1,P,P)
            out = model(patches).detach().to("cpu", torch.float64)
            for (r, c), o in zip(group, out):
                total[r:r + patch, c:c + patch] += o[0]
                count[r:r + patch, c:c + patch] += 1

    result = (total / count)[:H, :W]   # remove the padding again
    return result[None, None].to(device=x.device, dtype=torch.float32)


def denoise_slice(model, x, cfg):
    """Denoise one (1,1,H,W) slice in [0,1], choosing the right method:
    1. model.denoise(x)  if the model brings its own inference (e.g. diffusion)
    2. tiled(...)        if cfg["infer"]["patch"] is set
    3. model(x)          otherwise (the whole slice at once)
    """
    was_training = model.training
    model.eval()   # switch off dropout / batch-norm updates
    try:
        with torch.no_grad():
            if hasattr(model, "denoise"):
                return model.denoise(x)
            inf = cfg["infer"]
            if inf.get("patch"):
                return tiled(model, x, inf["patch"], inf.get("stride") or inf["patch"], inf.get("batch", 64))
            return model(x)
    finally:
        model.train(was_training)   # put the model back the way it was
