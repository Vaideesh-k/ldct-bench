"""Idea 2: CT-Mamba with CTformer-style attention in place of (or next to) some of
its Mamba blocks. Everything else is the official CT-Mamba (same loss, data, schedule).
Plain-language description and the shape map: docs/hybrid_design.md.

    model_args:
      variant: low | small | parallel     (see below)
      warmup_epochs: 10                   (loss switch, as in ct_mamba.py)

The three variants (configs/models/ct_mamba_hybrid.yaml + configs/variants/*.yaml):
  low       in the LOW-FREQUENCY branch (Net_low, the wavelet approximation band) every
            Mamba block (CZSS) is replaced by an attention block: the 32x32 block, the
            16x16 block (used twice), the 8x8 block (used 4 times) and the final 32x32
            fusion block. The three detail-band networks keep their Mamba blocks.
  small     in ALL FOUR band networks only the smallest-scale block (8x8 tokens, the
            one applied 4 times) is replaced by attention; everything else stays Mamba.
  parallel  in the low-frequency branch every Mamba block keeps its Mamba path and gets
            a parallel attention path; the two are mixed by a learnable scalar gate
            w = sigmoid(g) per block:  out = x + (1-w)*mamba(x) + w*attention(x),
            g initialised to 0 (w = 0.5).

THE ATTENTION BLOCK (T2TDilatedAttention) is built from CTformer's own modules
(third_party/CTformer, commit 3c62f58): a token-to-token step that gathers each
token's 3x3 DILATED neighbourhood (dilation 2, CTformer.py line 52: Unfold(3, dilation 2))
and squeezes it back to one token with CTformer's Token_performer (linear-attention
transformer layer, token_performer.py), a fixed sinusoid position code
(T2T_transformer_block.get_sinusoid_encoding), then one standard CTformer transformer
Block (8-head self-attention + MLP ratio 2, T2T_transformer_block.py lines 85-100,
the same Block CTformer uses as its bottleneck). Padding keeps the token grid, so the
block maps (B, H, W, C) -> (B, H, W, C) exactly like a CZSS block and is a drop-in.
Dropout/drop-path 0.1 as in CTformer.

Weight sharing is preserved: where CT-Mamba applies one CZSS module several times
(model.py lines 101-102, 107-110), the replacing attention module is applied the same
number of times.

Needs an NVIDIA GPU like CT-Mamba (the remaining Mamba blocks call .cuda()).
"""
import math
import os
import sys
import warnings

import torch
import torch.nn as nn

from ldct.config import ROOT
from ldct.models.ct_mamba import WARMUP_EPOCHS, CTMamba

CTFORMER = os.path.join(str(ROOT), "third_party", "CTformer")
VARIANTS = ("low", "small", "parallel")


def _ctformer_parts():
    """Token_performer, Block and the sinusoid table from the official CTformer code."""
    if not os.path.isdir(CTFORMER):
        raise SystemExit("third_party/CTformer not found. Get it with: bash scripts/get_ctformer.sh")
    if CTFORMER not in sys.path:
        sys.path.insert(0, CTFORMER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from T2T_transformer_block import Block, get_sinusoid_encoding
        from token_performer import Token_performer
    return Token_performer, Block, get_sinusoid_encoding


class T2TDilatedAttention(nn.Module):
    """CTformer-style attention block with the CZSS interface: (B, H, W, C) -> (B, H, W, C).

    inner(x) is the update, forward(x) = x + inner(x) (a residual block like CZSS)."""

    def __init__(self, dim, grid, heads=8, dilation=2, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        Token_performer, Block, sinusoid = _ctformer_parts()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.dim, self.grid = dim, grid
        self.unfold = nn.Unfold(kernel_size=3, dilation=dilation, padding=dilation)     # same grid out
        self.t2t = Token_performer(dim=9 * dim, in_dim=dim, kernel_ratio=0.5)            # 9 tokens -> 1
        self.register_buffer("pos", sinusoid(n_position=grid * grid, d_hid=dim), persistent=False)
        self.block = Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=False,
                           drop=drop, attn_drop=drop, drop_path=drop)
        self.norm = nn.LayerNorm(dim)

    def inner(self, x):
        B, H, W, C = x.shape
        t = self.unfold(x.permute(0, 3, 1, 2)).transpose(1, 2)        # (B, H*W, 9C): dilated neighbourhoods
        t = self.t2t(t)                                              # (B, H*W, C)
        if t.shape[1] == self.pos.shape[1]:
            t = t + self.pos
        t = self.norm(self.block(t))
        return t.reshape(B, H, W, C)

    def forward(self, x):
        return x + self.inner(x)


class ParallelBlock(nn.Module):
    """Mamba block and attention block side by side, mixed by a learnable gate."""

    def __init__(self, mamba_block, attention_block):
        super().__init__()
        self.mamba, self.attention = mamba_block, attention_block
        self.gate = nn.Parameter(torch.zeros(()))        # w = sigmoid(0) = 0.5 at the start

    def forward(self, x):
        w = torch.sigmoid(self.gate)
        return x + (1 - w) * (self.mamba(x) - x) + w * self.attention.inner(x)


# (attribute name inside MSC_Mamba, token-grid size for a 64x64 input) - model.py lines 50-86
LOW_BLOCKS = (("vssblock1", 32), ("vssblock2", 16), ("vssblock3", 8), ("vssblock", 32))
SMALL_BLOCKS = (("vssblock3", 8),)


def _swap(msc, blocks, make):
    """Replace CZSS modules of one MSC_Mamba module; returns the number swapped."""
    n = 0
    for name, grid in blocks:
        old = getattr(msc, name)
        dim = old.ln_1.normalized_shape[0]
        setattr(msc, name, make(old, dim, grid))
        n += 1
    return n


class CTMambaHybrid(CTMamba):
    def __init__(self, variant="low", warmup_epochs=WARMUP_EPOCHS):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        super().__init__(warmup_epochs=warmup_epochs)
        self.variant = variant
        net = self.net
        replace = lambda old, dim, grid: T2TDilatedAttention(dim, grid)                    # noqa: E731
        parallel = lambda old, dim, grid: ParallelBlock(old, T2TDilatedAttention(dim, grid))  # noqa: E731
        if variant == "low":
            self.swapped = _swap(net.Net_low.mambalayer, LOW_BLOCKS, replace)
        elif variant == "small":
            self.swapped = sum(_swap(m, SMALL_BLOCKS, replace)
                               for m in (net.Net_low.mambalayer, net.Net_H, net.Net_V, net.Net_D))
        else:
            self.swapped = _swap(net.Net_low.mambalayer, LOW_BLOCKS, parallel)

    def gates(self):
        """Current mixing weights of the parallel variant (for logging)."""
        return {n: float(torch.sigmoid(m.gate)) for n, m in self.named_modules() if isinstance(m, ParallelBlock)}


def build(variant="low", warmup_epochs=WARMUP_EPOCHS):
    return CTMambaHybrid(variant=variant, warmup_epochs=warmup_epochs)
