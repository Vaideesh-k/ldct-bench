"""CTformer (Wang et al., Physics in Medicine & Biology 2023), wrapped from the
official code so it follows our contract (see template.py).

Get the official code first:   bash scripts/get_ctformer.sh
It needs timm and einops (both in the ctmamba environment). It runs on CPU too.

All file/line references are to third_party/CTformer at commit 3c62f58
(the commit pinned in scripts/get_ctformer.sh).

NETWORK (solver.py line 71, the only configuration the official code ever builds):
  CTformer(img_size=64, tokens_type='performer', embed_dim=64, depth=1, num_heads=8,
           kernel=4, stride=4, mlp_ratio=2., token_dim=64)        -> 1,447,477 parameters
  (kernel/stride are accepted but unused by T2T_module; the test() in solver.py
  passes kernel=8, which therefore changes nothing.)
  Input: 64x64 patches only (the Fold sizes in Token_back_Image are fixed to 64,
  CTformer.py lines 101-103), so the config sets infer.patch: 64 for whole slices.
  The network predicts the noise and subtracts it (CTformer.py line 196).

VALUE RANGE: (HU + 1024) / 4096 (main.py lines 46-47), identical to our hu_to_unit.

TRAINING, as in the official main.py / solver.py / loader.py, reproduced via config:
  loss        MSE(pred, target) * 100 + 1e-4             solver.py line 135   -> compute_loss below
  optimiser   Adam, lr 1e-5                              main.py line 66, solver.py line 79
              (-> AdamW with weight_decay 0, which is Adam)
  patches     64x64, 4 per slice, 16 slices per step = 64 patches   main.py lines 52-54
              half the patches as cut, half flipped/rotated          loader.py lines 70-87
              (-> data.augment: true)
  data        Mayo 2016, test patient L506                main.py line 37
  schedule    official: 4000 epochs, lr halved every 8000 iterations (main.py lines 56-58,
              solver.py lines 149-150); the released checkpoint is from iteration 530,000
              (model_pretrained/). We train 1565 epochs x 339 steps = 530,000 steps on
              our 5,410 1-mm slices, so the number of updates matches the released model.

Differences from the official code (by design of this benchmark; see README):
  - data: 1 mm B30 slices (fixed rule). The official README uses the 3 mm set.
  - learning-rate schedule: our shared cosine decay (1e-5 -> 1e-7 over the run) instead
    of halving every 8000 iterations (which makes the official lr ~0 after ~150k steps).
  - final model: our fixed rule (epoch with the lowest training loss) instead of the
    last iteration.
  - full-slice inference: our tiles (64x64, stride 32) average where they overlap; the
    official test keeps the centre 32x32 of each tile (solver.py lines 25-45).
"""
import os
import sys

import torch.nn as nn
import torch.nn.functional as F

from ldct.config import ROOT

REPO = os.path.join(str(ROOT), "third_party", "CTformer")


def _import_official():
    if not os.path.isdir(REPO):
        raise SystemExit("third_party/CTformer not found. Get it with: bash scripts/get_ctformer.sh")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # timm's deprecation notices for old import paths
            from CTformer import CTformer
    except ImportError as e:
        raise SystemExit(f"Could not import the official CTformer code: {e}. "
                         f"It needs timm and einops (pip install timm einops).")
    return CTformer


class CTformerModel(nn.Module):
    """Official CTformer network plus its training loss."""

    def __init__(self):
        super().__init__()
        CTformer = _import_official()
        # exactly solver.py line 71
        self.net = CTformer(img_size=64, tokens_type="performer", embed_dim=64, depth=1,
                            num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, pred, target, inp):
        # solver.py line 135: MSE * 100 + 1e-4 ("to prevent 0")
        return F.mse_loss(pred, target) * 100 + 1e-4


def build():
    return CTformerModel()
