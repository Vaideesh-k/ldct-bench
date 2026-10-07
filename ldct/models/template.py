"""TEMPLATE for adding a model. Copy it and fill it in:

    cp ldct/models/template.py ldct/models/<your_model>.py
    (optional) create configs/models/<your_model>.yaml for its own settings
    python scripts/train.py --model <your_model> --smoke

THE CONTRACT
------------
Input and output are images shaped (B, 1, H, W) with values in [0, 1]
(HU scaled by ldct.data.hu_to_unit: 0 = -1024 HU, 1 = 3072 HU).
Training gives the model 64x64 patches; testing gives whole 512x512 slices
(or 64x64 tiles if your config sets infer.patch, see ldct/infer.py).

Required:
    build(**model_args) -> nn.Module
        model_args come from configs/models/<name>.yaml, e.g.
            model_args: {channels: 96}

Optional hooks: the pipeline uses them automatically if your model has them.
    compute_loss(pred, target, inp) -> scalar tensor
        Your own training loss. Default when missing: L1(pred, target).
    training_step(inp, target) -> scalar tensor
        Replaces the whole "pred = model(inp); loss = ..." step.
        For diffusion models, whose training is not "predict the clean image".
    denoise(x) -> tensor shaped like x
        Your own inference for one (1,1,H,W) slice, e.g. a diffusion sampling loop.
        Default when missing: model(x), or tiles if infer.patch is set.
Before every epoch, scripts/train.py sets model.epoch (1, 2, 3, ...) so a loss
can change during training (CT-Mamba does this after epoch 10).

WRAPPING AN OFFICIAL REPO
-------------------------
Do not copy the official code into ldct/. Instead:
1. Add scripts/get_<name>.sh that clones the official repo into
   third_party/<Repo> (third_party/ is git-ignored), pinned to one commit.
2. In ldct/models/<name>.py add that folder to sys.path and import the class:
       REPO = os.path.join(ROOT, "third_party", "<Repo>")
       sys.path.insert(0, REPO)
       from model import TheirNet
3. Wrap it in a small nn.Module that follows the contract above (convert the
   value range if theirs is not [0, 1], put their loss in compute_loss).
See ldct/models/ct_mamba.py for a complete example.
"""
import torch.nn as nn


class TinyDenoiser(nn.Module):
    """Example model: a few conv layers that predict the noise and subtract it."""

    def __init__(self, channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x):
        return x - self.body(x)   # noisy image minus the predicted noise

    # Optional hook example (delete it to use the default L1 loss):
    # def compute_loss(self, pred, target, inp):
    #     return torch.nn.functional.mse_loss(pred, target)


def build(channels=32):
    """Called by ldct.models.build_model with the model_args from the config."""
    return TinyDenoiser(channels=channels)
