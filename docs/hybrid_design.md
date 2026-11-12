# Idea 2: a CT-Mamba / CTformer hybrid

Plain-language design notes for `ldct/models/ct_mamba_hybrid.py`. All line numbers refer
to the pinned official repos (`third_party/CT-Mamba` at 89227e1, `third_party/CTformer`
at 3c62f58). Shapes are for one training patch of 64x64 pixels; B is the batch.

## 1. What CT-Mamba does with a patch

```
input (B,1,64,64)
 └─ PFEN: a few convolutions                            -> (B,32,64,64)        model.py 196-221
 └─ Haar wavelet, one level (ptwt.wavedec2)             -> 4 bands (B,32,32,32)  model.py 290
      cA  low-frequency "approximation" (the smooth image)
      cH, cV, cD  horizontal / vertical / diagonal details (edges, noise)
 ├─ LOW-FREQUENCY BRANCH  Net_low(cA)                    -> (B,32,32,32)         model.py 165-190
 │    split 16+16 channels: FFTNet (frequency-domain convs) | ResNet10
 │    concat (B,32,32,32) -> MSC_Mamba(in 32, dims 128)
 ├─ DETAIL BRANCHES (each a MSC_Mamba, fed with its band plus "fused" neighbours)
 │    Net_H( cat[cH, HFEN(cD)] )  = MSC_Mamba(in 64,  dims 256)
 │    Net_V( cat[cV, HFEN(cD)] )  = MSC_Mamba(in 64,  dims 256)
 │    Net_D( cat[cD, HFEN(cV), HFEN(cH)] ) = MSC_Mamba(in 96, dims 384)
 └─ inverse wavelet (ptwt.waverec2) of the 4 outputs      -> (B,32,64,64)
    + residual, PFFN convolutions                          -> (B,1,64,64)
```

### MSC_Mamba: the multi-scale Mamba module (model.py 46-120)

Input `(B,C,32,32)` (C = 32, 64 or 96), output always `(B,32,32,32)`.

| step | module | tokens | channels | where |
|---|---|---|---|---|
| 0 | ResBlock (2 convs) | 32x32 | C | |
| 1 | `vssblock1` = CZSS (Z-scan Mamba block) | 32x32 | C | scale 1 |
| 2 | PatchEmbed (2x2) -> `vssblock2` CZSS **applied twice** -> expand back | 16x16 | D = 4C | scale 2 |
| 3 | PatchEmbed (4x4) -> `vssblock3` CZSS **applied 4 times** -> expand back | 8x8 | 4D = 16C | scale 3 |
| 4 | 3 x (conv C -> 32), concat -> 96 ch, 2 convs -> 32 ch | 32x32 | 32 | fusion |
| 5 | `vssblock` = CZSS | 32x32 | 32 | fusion |

So every band network has **3 scales** (32x32, 16x16, 8x8 tokens) and 4 distinct Mamba
blocks (1 + 1 + 1 + 1, two of them shared across repeated applications).

### CZSS: one Z-scan Mamba block (mamba.py 156-395)

`x (B,H,W,C) -> x + DropPath( Z_SS2D( LayerNorm(x) ) )`

Z_SS2D: Linear C -> 2*2C (split into the signal `x` and a gate `z`); depth-wise 3x3 conv +
SiLU on `x`; the H*W tokens are flattened in **four orders** - a Z-order (Morton) curve
over the grid, the same over the transposed grid, and both reversed (the scan tables
`index/<grid>_ZScan.npy`, which is why the grid size is fixed to 32/16/8); a selective
state-space scan (`selective_scan_fn`, state size 16) runs along each order; the four
results are added, LayerNorm-ed, multiplied by SiLU(z), and projected back 2C -> C.
Cost grows linearly with the number of tokens; each token only "sees" tokens before it
in each of the 4 scan orders.

## 2. What CTformer does with a patch (CTformer.py, solver.py 71)

```
input (B,1,64,64)
 └─ T2T module A: Unfold 7x7 stride 2  -> 29x29 = 841 tokens of 49 values
                  Token_performer 49 -> 64                          (B,841,64)
 └─ roll by 2, reshape to 29x29, Unfold 3x3 DILATION 2 -> 25x25 = 625 tokens of 9*64
                  Token_performer 576 -> 64                          (B,625,64)   module B
 └─ roll, Unfold 3x3 -> 23x23 = 529 tokens of 576, Linear -> 64      (B,529,64)
 └─ + sinusoid position code; 1 transformer Block (8 heads, MLP x2)  (B,529,64)
 └─ mirror image of the above with Fold instead of Unfold (modules C, D) -> (B,1,64,64)
 output = input - decoded                                                   (noise is subtracted)
```

The "token-to-token dilated attention block" is the pair *gather a 3x3 dilated
neighbourhood of tokens into one long token* -> *transformer layer over all tokens*
(module B); the dilation lets each new token cover a wider area without more
parameters. Token_performer is a linear-attention ("performer") transformer layer:
LayerNorm, kqv projection, random-feature softmax kernel with m = C/2 features,
v + proj(attention) skip, then an MLP; its output has `in_dim` channels, which is how
the 9C-wide gathered token becomes a C-wide token again. The Block is a plain
pre-norm transformer layer with exact 8-head attention: every token sees every token.

## 3. The drop-in attention block used here

`T2TDilatedAttention(dim=C, grid=G)` keeps the CZSS interface `(B,G,G,C) -> (B,G,G,C)`:

| step | op | shape |
|---|---|---|
| gather | `Unfold(3x3, dilation 2, padding 2)` on `(B,C,G,G)` | `(B, 9C, G*G)` -> transpose -> `(B, G*G, 9C)` |
| token-to-token | CTformer `Token_performer(dim=9C, in_dim=C)` | `(B, G*G, C)` |
| position | + fixed sinusoid table for G*G positions | |
| attention | CTformer `Block(dim=C, heads=8, mlp_ratio=2)` + LayerNorm | `(B, G*G, C)` |
| out | reshape, residual: `x + update` | `(B, G, G, C)` |

Differences from CTformer's own use of these modules: no stride-2 unfold and no
roll (the grid must stay the same size to replace a CZSS block), padding keeps the
grid, dims follow the CT-Mamba branch (C = 32..512) instead of 64. Dropout 0.1 as
in CTformer. Heads = 8 divides every C that occurs (32, 64, 96, 128, 256, 384, 512).

## 4. The three variants

| variant | what changes | blocks swapped | why try it |
|---|---|---|---|
| (a) `low` | every CZSS in the low-frequency branch (`Net_low.mambalayer`: vssblock1, vssblock2 x2 uses, vssblock3 x4 uses, vssblock) becomes attention | 4 modules | the smooth band carries the organ contrast; global attention may model long-range structure (liver, lesions) better than directional scans |
| (b) `small` | only `vssblock3` (8x8 tokens, the cheapest scale) in all four band networks becomes attention | 4 modules | the smallest grid has only 64 tokens, so exact attention is almost free; tests whether global context at coarse scale is what helps |
| (c) `parallel` | in the low-frequency branch each CZSS keeps its Mamba path and gains a parallel attention path; `out = x + (1-w)*mamba_update + w*attention_update`, `w = sigmoid(g)`, g learnable, init 0 | 4 modules wrapped | lets training decide how much attention to use; the learned w values (logged by `model.gates()`) are themselves a result |

Everything else (loss, data, schedule, best.pth rule, tiles at test time) is the official
CT-Mamba via `CTMamba`, which the hybrid subclasses. Weight sharing across repeated
applications is kept. The parameter counts, memory at batch 32 and seconds per step of
all four models are printed by `tests/check_hybrid.py` (GPU part; job script
`slurm/hybrid_test.sbatch`) and recorded in docs/PROGRESS.md once it has run.

## 5. The short-run comparison (not submitted; see slurm/short50.sbatch)

Four 50-epoch runs with identical settings (`configs/short50.yaml`: cosine over 50
epochs, loss switch at epoch 3 instead of 11, best.pth among epochs >= 3), one after
another on one A100: plain CT-Mamba, then the three variants. Scored on L506 with the
shared scorer. A 50-epoch CT-Mamba is only a reference point for the variants, never
a replacement for the 250-epoch result.
