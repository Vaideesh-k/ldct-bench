"""CoreDiff (Gao et al., IEEE TMI 2024), wrapped from the official code so it
follows our contract (see template.py). A cold-diffusion model: 10 steps from the
low-dose image to the clean one, with a small U-Net; it trains and tests on whole
512x512 slices and takes the slice above and below as extra input channels.

Get the official code first:   bash scripts/get_corediff.sh
It needs einops (in the ctmamba environment). The network itself runs on CPU too.

All file/line references are to third_party/CoreDiff at commit 9bc31dc
(the commit pinned in scripts/get_corediff.sh). Only two official files are used,
models/corediff/corediff_wrapper.py (the U-Net) and models/corediff/diffusion_modules.py
(the diffusion process); they are loaded by file so that the official package
__init__ (which needs wandb) is never imported.

SETTINGS = the official Mayo 2016 run (train_mayo2016.sh + the defaults in
models/basic_template.py lines 24-73 and models/corediff/corediff.py lines 22-37):
  network      Network(in_channels=3, context=True) = UNet(3 -> 1) + EMM "adjust" nets
  diffusion    Diffusion(image_size=512, timesteps=T=10, context=True)
  input        3 channels: low-dose slices i-1, i, i+1 (utils/dataset.py lines 33-40)
               -> data.context: 1 in the config
  loss         0.5*MSE(x_recon, y) + 0.5*MSE(x_recon_sub1, y)   corediff.py line 103
  optimiser    Adam, lr 2e-4, constant (adjust_learning_rate is empty)  corediff.py line 57
               -> AdamW with weight_decay 0 and lr_min = lr
  batch        8 whole slices per step (train_mayo2016.sh --batch_size 8)
  iterations   150,000 (--max_iter)  -> 222 epochs x 677 steps = 150,294 here
  EMA          decay 0.995, updated every 10 steps, reset to the model before step 2000
               (corediff.py lines 24-26, 69-74, 108-109); testing uses the EMA model
               with 10-step 'ddim' sampling (corediff.py lines 112-125)
  flags        --context --only_adjust_two_step, start_adjust_iter 1, dose 25 (= quarter)
  data         prep_mayo2016.py reads the 1 mm full_1mm / quarter_1mm DICOMs: the same
               slices as ours; test patient index 9 = L506.

VALUE RANGE: (HU + 1024) / 4096 after clipping HU to [-1024, 3072]
(utils/dataset.py lines 141-146) = our hu_to_unit (we do not clip; values above
3072 HU are metal and do not occur in these patients).

Differences from the official code (by design of this benchmark; see README):
  - final model: our fixed rule (EMA weights of the epoch with the lowest training
    loss) instead of the EMA weights at iteration 150,000.
  - the first and last slice of every patient: the official loader drops them
    (dataset.py lines 32, 38); we keep them and repeat the edge slice as the missing
    neighbour (ldct.data.neighbours), so all 526 L506 slices are scored like for
    every other model.
  - the EMA update for step n happens at the start of step n+1 (our training hook
    runs before the optimiser step), so the EMA saved at an epoch end can be up to
    9 steps behind the official timing.
"""
import copy
import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldct.config import ROOT

REPO = os.path.join(str(ROOT), "third_party", "CoreDiff")


def _load(name):
    """Import one official file by path (bypasses models/__init__.py, which needs wandb)."""
    path = os.path.join(REPO, "models", "corediff", f"{name}.py")
    if not os.path.exists(path):
        raise SystemExit("third_party/CoreDiff not found. Get it with: bash scripts/get_corediff.sh")
    spec = importlib.util.spec_from_file_location(f"corediff_official_{name}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        raise SystemExit(f"Could not import the official CoreDiff code: {e} (pip install einops).")
    return module


class CoreDiffModel(nn.Module):
    """Official CoreDiff network + diffusion process, its EMA copy and its training loss."""

    loss_modules = ("ema",)   # the EMA copy is not trained by the optimiser; counted separately

    def __init__(self, image_size=512, T=10, context=True, ema_decay=0.995,
                 update_ema_every=10, start_ema_step=2000, only_adjust_two_step=True, start_adjust_iter=1):
        super().__init__()
        wrapper, diffusion = _load("corediff_wrapper"), _load("diffusion_modules")
        in_channels = 3 if context else 1
        # corediff.py lines 46-54 (set_model)
        self.net = diffusion.Diffusion(denoise_fn=wrapper.Network(in_channels=in_channels, context=context),
                                       image_size=image_size, timesteps=T, context=context)
        self.ema = copy.deepcopy(self.net)
        for p in self.ema.parameters():
            p.requires_grad = False
        self.T = T
        self.ema_decay, self.update_ema_every, self.start_ema_step = ema_decay, update_ema_every, start_ema_step
        self.only_adjust_two_step, self.start_adjust_iter = only_adjust_two_step, start_adjust_iter
        # completed optimiser steps; a buffer so --resume continues the count
        self.register_buffer("n_iter", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        """Plain forward = denoise (10 sampling steps with the EMA model)."""
        return self.denoise(x)

    @torch.no_grad()
    def _ema_step(self, n):
        """corediff.py lines 69-74 and 108-109, for completed step n."""
        if n == 0 or n % self.update_ema_every:
            return
        if n < self.start_ema_step:
            self.ema.load_state_dict(self.net.state_dict())       # reset_parameters()
            return
        for p_ema, p in zip(self.ema.parameters(), self.net.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def training_step(self, inp, target):
        """One official training step (corediff.py lines 86-103): inp (B,3,H,W), target (B,1,H,W)."""
        self._ema_step(int(self.n_iter))          # EMA update owed by the previous step
        n_iter = int(self.n_iter) + 1
        x_recon, _, x_recon_sub1, _ = self.net(inp, target, n_iter,
                                               only_adjust_two_step=self.only_adjust_two_step,
                                               start_adjust_iter=self.start_adjust_iter)
        loss = 0.5 * F.mse_loss(x_recon, target) + 0.5 * F.mse_loss(x_recon_sub1, target)
        self.n_iter += 1
        return loss

    @torch.no_grad()
    def denoise(self, x):
        """corediff.py lines 112-125: 10-step ddim sampling with the EMA model; x is (1,3,H,W)."""
        out, _, _ = self.ema.sample(batch_size=x.shape[0], img=x, t=self.T, sampling_routine="ddim",
                                    n_iter=max(int(self.n_iter), self.start_adjust_iter),
                                    start_adjust_iter=self.start_adjust_iter)
        return out


def build(**model_args):
    return CoreDiffModel(**model_args)
