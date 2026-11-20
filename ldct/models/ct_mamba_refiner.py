"""CT-Mamba refiner (final-week step 3): frozen CT-Mamba (Stage A) + a small 2D U-Net (Stage B)
that adds a correction to the Stage-A image.

    input  (B, 2, H, W) in [0, 1]: channel 0 = noisy quarter dose, channel 1 = Stage-A output
           2.5D variant (in_channels=4): [noisy z-1, noisy z, noisy z+1, Stage A]
    output (B, 1, H, W) = Stage-A output (always the LAST input channel) + U-Net residual

Stage B: 4 levels with 32/64/128/256 channels, each level 2 x (3x3 conv + GroupNorm + SiLU),
max-pool down, bilinear upsample + 3x3 conv up, skip connections (concatenation), a final
1x1 conv to one channel that is ZERO-INITIALISED, so the untrained refiner returns the
Stage-A image exactly.

Stage A is not inside this module: CT-Mamba is frozen, so its outputs are computed once
(scripts/precompute_stage_a.py for training, the CT-Mamba predictions for testing). At
inference scripts/predict_refiner.py runs CT-Mamba, then this module (or reads the identical
precomputed CT-Mamba output). H and W must be divisible by 8 (128 patches, 512 slices).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, groups=8):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.block = _block(2 * cout, cout)

    def forward(self, x, skip):
        x = self.conv(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
        return self.block(torch.cat([x, skip], 1))


class Refiner(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), in_channels=2):
        super().__init__()
        c = list(channels)
        self.in_channels = in_channels
        self.downs = nn.ModuleList([_block(in_channels, c[0])] +
                                   [_block(c[i - 1], c[i]) for i in range(1, len(c))])
        self.ups = nn.ModuleList([Up(c[i], c[i - 1]) for i in range(len(c) - 1, 0, -1)])
        self.head = nn.Conv2d(c[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"the refiner takes {self.in_channels} channels [noisy slice(s), Stage A], got {x.shape[1]}")
        return x[:, -1:] + self.residual(x)


def build(channels=(32, 64, 128, 256), in_channels=2):
    return Refiner(tuple(channels), in_channels)
