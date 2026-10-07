"""CT-Mamba (Li et al., Computerized Medical Imaging and Graphics 2025), wrapped
from the official code so it follows our contract (see template.py).

Get the official code first:   bash scripts/get_ct_mamba.sh
It needs an NVIDIA GPU: the official code calls .cuda() everywhere and its
Mamba layers need mamba-ssm (plus causal-conv1d, timm, einops, PyWavelets,
ptwt, torchvision). It cannot run on a Mac.

All file/line references below are to third_party/CT-Mamba at commit 89227e1
(the commit pinned in scripts/get_ct_mamba.sh).

Value range: the official data is (HU + 1024) / 4096 (official README), which is
exactly our hu_to_unit, so images go in and out unchanged.
Input size: the scan-order tables are fixed to 64x64 patches (mamba.py
lines 197-215), so the config sets infer.patch: 64 to denoise whole slices in tiles.

TRAINING LOSS (reproduced from train/train_denoise.py)
  epochs 1-10  (lines 208-209): NPSloss1
  epochs 11+   (lines 210-211): NPSloss, the "Deep NPS loss"
Both share the same terms (NPSloss.py lines 293-308):
  loss1 = 1.0  * L1(pred, target) * 100 + 1e-4                    main term
  loss2 = 1e-4 * L1(radial NPS of pred noise, of target noise)    noise texture
  loss3 = 0.01 * (1 - Pearson(same two radial NPS curves))        noise texture
  loss4 = 0.01 * MSE of ResNet-50 features (4 blocks, averaged)   perceptual
NPSloss1 sets the loss2/loss3 weights to 0.0 (NPSloss.py lines 318-319), so for
epochs 1-10 only loss1 + loss4 count; we compute exactly those two terms.
The noise feature nets (unet_feature, unet_feature1, ss2d, ss2d1) are trained
together with the model (train_denoise.py lines 77-86); here they live inside
this module, so our optimizer trains them too and they are saved in checkpoints.
pixelSpacing is 1.0 (train_denoise.py line 205).

NEEDS PRETRAINED WEIGHTS: loss4 uses torchvision's ImageNet ResNet-50
(NPSloss.py line 252, pretrained=True). torchvision downloads it (~100 MB)
the first time, so the first run needs internet. It stays frozen and in eval
mode (as in the official code, where it is never trained: NPSloss.py line 255,
train_denoise.py lines 77-86).

Differences from the official script (by design of this benchmark):
  - learning rate: our plain cosine schedule; the official one adds a 3-epoch
    warm-up first (train_denoise.py lines 97-102, optionsmamba.py lines 42-43).
  - full-slice inference: our tiles average where they overlap; the official
    code keeps only the centre 32x32 of each 64x64 tile (train_denoise.py lines 163-182).
"""
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

from ldct.config import ROOT

REPO = os.path.join(str(ROOT), "third_party", "CT-Mamba")
WARMUP_EPOCHS = 10        # NPSloss1 for epochs 1-10 (train_denoise.py line 208)
PIXEL_SPACING = 1.0       # train_denoise.py line 205


@contextmanager
def _inside_repo():
    """The official code loads index/*.npy with paths relative to its own folder
    (mamba.py lines 197-215), so its modules are built from inside that folder."""
    old = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(old)


def _import_official():
    """Import CT_Mamba (model.py) and the loss module (NPSloss.py), with clear errors."""
    if not os.path.isdir(REPO):
        raise SystemExit("third_party/CT-Mamba not found. Get it with: bash scripts/get_ct_mamba.sh")
    if not torch.cuda.is_available():
        raise SystemExit("CT-Mamba needs an NVIDIA GPU with CUDA (the official code calls .cuda()). "
                         "It cannot run on a Mac or CPU.")
    try:
        import mamba_ssm  # noqa: F401  (the official mamba.py hides this error, so check it here)
    except ImportError as e:
        raise SystemExit(f"mamba-ssm is not installed in this environment ({e}). "
                         f"Activate the ctmamba conda environment (see LAPTOP_RUN.md).")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from model import CT_Mamba
        import NPSloss
    except ImportError as e:
        raise SystemExit(f"Could not import the official CT-Mamba code: {e}. "
                         f"A package is probably missing from this environment (see LAPTOP_RUN.md).")
    return CT_Mamba, NPSloss


class CTMamba(nn.Module):
    """Official CT_Mamba network plus its Deep NPS training loss."""

    loss_modules = ("nps",)   # not part of the denoiser; train.py counts it separately

    def __init__(self):
        super().__init__()
        CT_Mamba, NPSloss = _import_official()
        with _inside_repo():
            self.net = CT_Mamba()            # model.py line 264; built like utils/model_utils.py line 62
            self.nps = NPSloss.NPSloss()     # NPSloss.py lines 233-261, default weights
        # the pretrained ResNet-50 is only a judge: never trained (not in the
        # official optimizer, train_denoise.py lines 77-86)
        for p in self.nps.model.parameters():
            p.requires_grad = False
        self.epoch = 1                       # set by scripts/train.py before every epoch

    def forward(self, x):
        return self.net(x)

    def train(self, mode=True):
        """Switch train/eval mode, but keep the ResNet-50 in eval mode always
        (NPSloss.py line 255; the official script never switches it)."""
        super().train(mode)
        self.nps.model.eval()
        return self

    def _feature_loss(self, pred, target):
        """Mean MSE between ResNet-50 features of pred and target (NPSloss.py lines 297-304)."""
        pred_feats = self.nps.model(torch.cat([pred, pred, pred], dim=1))
        target_feats = self.nps.model(torch.cat([target, target, target], dim=1))
        total = 0
        for p_f, t_f in zip(pred_feats, target_feats):
            total = total + self.nps.mse_loss(p_f, t_f)
        return total / len(self.nps.blocks)

    def compute_loss(self, pred, target, inp):
        if self.epoch <= WARMUP_EPOCHS:
            # NPSloss1: radial weights are 0.0, so only loss1 + loss4 (NPSloss.py lines 371, 375-386)
            loss1 = self.nps.mae_weight * self.nps.mae_loss(pred, target) * 100 + 1e-4
            loss4 = self.nps.feature_weight * self._feature_loss(pred, target)
            return loss1 + loss4
        # full Deep NPS loss, called exactly like train_denoise.py line 211
        loss, _, _, _, _ = self.nps(inp, pred, target, PIXEL_SPACING)
        return loss


def build():
    return CTMamba()
