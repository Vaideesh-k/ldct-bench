"""Final architecture figure: frozen CT-Mamba (Stage A) + 2D U-Net refiner (Stage B), and how it is trained."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

NAVY, TEAL, ORANGE, SLATE, LIGHT, INK, RED = "#0F2A3D", "#0E8A96", "#E3893B", "#5B6F7C", "#EEF3F5", "#14232E", "#B8443A"
TEAL_L, ORANGE_L = "#DCEFF1", "#FBE9D8"

fig = plt.figure(figsize=(16, 10), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 62.5); ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, fc, ec=None, r=0.6, lw=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw, zorder=z))


def text(x, y, s, size=11, color=INK, weight="normal", ha="center", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def arrow(p, q, color=SLATE, lw=1.8, style="-|>", ls="-", z=3, ms=14):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def line(xs, ys, color=SLATE, lw=1.8, ls="-", z=3):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


# ------------------------------------------------------------------ title
text(1.5, 60.7, "Final architecture: CT-Mamba + U-Net refiner", 22, NAVY, "bold", ha="left", family="serif")
text(1.5, 58.9, "Stage A removes the noise (frozen CT-Mamba). Stage B, a small U-Net, learns a correction that "
     "sharpens tumour edges. It is trained with a tumour-weighted loss.", 11.5, SLATE, ha="left")

ROW = 49.5   # centre line of the main data path

# ------------------------------------------------------------------ input
box(1.5, ROW - 3.2, 8.5, 6.4, SLATE)
text(5.75, ROW + 1.0, "Quarter-dose\nCT slice", 11.5, "white", "bold")
text(5.75, ROW - 2.0, "512 × 512, HU", 9.5, "#DDE5E9")

# ------------------------------------------------------------------ stage A
box(12.5, 36.5, 21.5, 18.3, NAVY)
text(23.25, 52.9, "Stage A: CT-Mamba", 13, "white", "bold")
text(23.25, 51.0, "FROZEN, not trained (best checkpoint, epoch 237)", 9, "#9CC9CF")
inner = ["PFEN feature extractor", "Haar wavelet → 4 frequency bands", "Z-scan Mamba blocks, 3 scales per band", "Inverse wavelet → denoised slice"]
for i, s in enumerate(inner):
    y = 47.6 - i * 2.75
    box(14.0, y - 1.05, 18.5, 2.1, "#1F4560", r=0.35, z=3)
    text(23.25, y, s, 9.8, "white")
    if i < len(inner) - 1:
        arrow((23.25, y - 1.05), (23.25, y - 1.7), "#9CC9CF", 1.2, ms=9)
text(23.25, 37.5, "50.8 M parameters", 9, "#9CC9CF")
arrow((10.0, ROW), (12.5, ROW))

# stage A output
box(36.0, ROW - 3.2, 7.6, 6.4, NAVY)
text(39.8, ROW + 0.9, "Stage-A\nimage", 11.5, "white", "bold")
text(39.8, ROW - 2.1, "denoised", 9.5, "#9CC9CF")
arrow((34.0, ROW), (36.0, ROW))

# concat
CX = 46.6
ax.add_patch(Circle((CX, ROW), 1.3, fc="white", ec=SLATE, lw=1.8, zorder=4))
text(CX, ROW, "C", 12, SLATE, "bold")
text(CX, ROW + 2.4, "concat", 9, SLATE)
arrow((43.6, ROW), (CX - 1.3, ROW))
# noisy input routed below stage A into the concat
line([5.75, 5.75, CX], [ROW - 3.2, 34.3, 34.3])
arrow((CX, 34.3), (CX, ROW - 1.3))
text(26, 33.4, "the noisy slice is also given to the U-Net", 9, SLATE, style="italic")
text(45.9, 44.3, "2 channels\n[noisy, Stage A]", 8.5, SLATE, ha="right")

# ------------------------------------------------------------------ stage B: U-Net
ax.add_patch(FancyBboxPatch((49.3, 29.6), 38.4, 26.6, boxstyle="round,pad=0,rounding_size=0.8", fc=TEAL_L, ec=TEAL, lw=1.4, zorder=1))
text(68.5, 54.6, "Stage B: 2D U-Net refiner (TRAINED, 2.14 M parameters)", 12.5, TEAL, "bold")

BW, BH = 7.6, 3.6
enc = [(50.8, ROW - BH / 2, "32 ch", "128²"), (52.8, 43.0 - BH / 2, "64 ch", "64²"), (54.8, 36.8 - BH / 2, "128 ch", "32²")]
dec = [(77.6, ROW - BH / 2, "32 ch", "128²"), (75.6, 43.0 - BH / 2, "64 ch", "64²"), (73.6, 36.8 - BH / 2, "128 ch", "32²")]
bott = (63.1, 30.7, 9.6, BH)
for (x, y, c, s) in enc:
    box(x, y, BW, BH, TEAL, r=0.4, z=3)
    text(x + BW / 2, y + BH / 2 + 0.6, c, 10.5, "white", "bold"); text(x + BW / 2, y + BH / 2 - 0.9, s, 9, "#DDF1F3")
for (x, y, c, s) in dec:
    box(x, y, BW, BH, "#0B6E78", r=0.4, z=3)
    text(x + BW / 2, y + BH / 2 + 0.6, c, 10.5, "white", "bold"); text(x + BW / 2, y + BH / 2 - 0.9, s, 9, "#DDF1F3")
box(*bott, NAVY, r=0.4, z=3)
text(bott[0] + bott[2] / 2, bott[1] + BH / 2 + 0.6, "256 ch", 10.5, "white", "bold")
text(bott[0] + bott[2] / 2, bott[1] + BH / 2 - 0.9, "16² (bottleneck)", 9, "#9CC9CF")
arrow((CX + 1.3, ROW), (50.8, ROW))

# down path (max-pool) and up path (bilinear up + 3x3 conv)
downs = [(enc[0][0] + 2.0, enc[0][1], enc[1][0] + 2.0, enc[1][1] + BH),
         (enc[1][0] + 2.0, enc[1][1], enc[2][0] + 2.0, enc[2][1] + BH),
         (enc[2][0] + 3.0, enc[2][1], bott[0] + 1.0, bott[1] + BH / 2)]
for i, (x1, y1, x2, y2) in enumerate(downs):
    if i < 2:
        arrow((x1, y1), (x2, y2), ORANGE, 1.8)
    else:
        line([x1, x1], [y1, y2], ORANGE); arrow((x1, y2), (x2, y2), ORANGE, 1.8)
ups = [(bott[0] + bott[2] - 1.0, bott[1] + BH / 2, dec[2][0] + BW - 3.0, dec[2][1]),
       (dec[2][0] + BW - 2.0, dec[2][1] + BH, dec[1][0] + BW - 2.0, dec[1][1]),
       (dec[1][0] + BW - 2.0, dec[1][1] + BH, dec[0][0] + BW - 2.0, dec[0][1])]
for i, (x1, y1, x2, y2) in enumerate(ups):
    if i == 0:
        line([x1, x2], [y1, y1], NAVY); arrow((x2, y1), (x2, y2), NAVY, 1.8)
    else:
        arrow((x1, y1), (x2, y2), NAVY, 1.8)
text(57.2, 33.3, "max-pool ↓", 9, ORANGE, "bold", ha="right")
text(79.0, 33.0, "↑ bilinear up\n   + 3×3 conv", 9, NAVY, "bold", ha="left")
# skips
for (ex, ey, _, _), (dx, dy, _, _) in zip(enc, dec):
    arrow((ex + BW, ey + BH / 2), (dx, dy + BH / 2), SLATE, 1.3, ls=(0, (4, 3)))
text(68.5, ROW + 0.8, "skip connections (concatenate)", 8.8, SLATE, style="italic")
text(68.5, 40.6, "each block = 2 × (3×3 conv → GroupNorm → SiLU)", 8.8, INK)

# head + residual add
box(88.6, ROW - 2.4, 5.0, 4.8, ORANGE, r=0.4, z=3)
text(91.1, ROW + 0.8, "1×1 conv", 9.8, "white", "bold"); text(91.1, ROW - 1.0, "zero-init", 8.8, "white")
arrow((dec[0][0] + BW, ROW), (88.6, ROW))
text(91.7, ROW - 4.3, "residual r", 8.5, ORANGE, "bold", ha="left")
ax.add_patch(Circle((91.1, 41.0), 1.3, fc="white", ec=ORANGE, lw=1.8, zorder=4))
text(91.1, 41.0, "+", 16, ORANGE, "bold")
arrow((91.1, ROW - 2.4), (91.1, 42.3), ORANGE)
# stage A image skip to the adder: above everything
line([39.8, 39.8, 96.2, 96.2], [ROW + 3.2, 56.9, 56.9, 41.0], NAVY, 1.6)
arrow((96.2, 41.0), (92.4, 41.0), NAVY, 1.6)
text(70.0, 57.2, "Stage-A image is added back:  output = Stage A + r", 9.5, NAVY, "bold", va="bottom")
# output
box(88.0, 30.0, 10.4, 6.8, NAVY)
text(93.2, 34.4, "Refined slice", 11.5, "white", "bold")
text(93.2, 31.9, "denoised +\nsharper tumour edges", 8.5, "#9CC9CF")
arrow((91.1, 39.7), (91.1, 36.8))
text(50.5, 27.6, "At test time only the CT image goes in: no tumour masks and no nnU-Net are used. "
     "The zero-init head means training starts exactly at CT-Mamba's output.", 10, INK, ha="center", style="italic")

# ------------------------------------------------------------------ training
text(1.5, 24.6, "How we train it", 17, NAVY, "bold", ha="left", family="serif")
cards = [
    ("1  Data", TEAL, [
        "128 × 128 patches, batch 16",
        "50% Mayo: 9 train patients,",
        "   real quarter / normal dose pairs",
        "50% LiTS: 41 train patients,",
        "   simulated quarter dose",
        "~30% of LiTS patches centred",
        "   on a tumour",
        "Test patients never used"]),
    ("2  Loss", ORANGE, [
        "L = mean|ŷ − y|",
        "     + 4 · mean(M · |ŷ − y|)",
        "M = tumour mask, dilated 3 px",
        "→ tumour pixels weigh 5×",
        "y = normal-dose slice",
        "Masks are used ONLY in",
        "   the loss, never at test time"]),
    ("3  Optimisation", NAVY, [
        "Only the U-Net is updated",
        "CT-Mamba frozen; its outputs",
        "   precomputed once (int16)",
        "AdamW, lr 2e-4 → 1e-6 (cosine)",
        "weight decay 1e-4",
        "30,000 iterations",
        "Checkpoint = lowest 1,000-",
        "   iteration training-loss window"]),
    ("4  Fair control", SLATE, [
        "CT-Mamba itself fine-tuned",
        "   with the SAME data, sampling",
        "   and loss",
        "Same pixels per step:",
        "   16 × 128² = 64 × 64²",
        "   = 262,144",
        "DONE: 30,000 iterations,",
        "   best loss 0.004045"]),
    ("5  How we judge it", RED, [
        "Mayo L506: PSNR ≥ 29.42 dB",
        "LiTS: same fixed nnU-Net →",
        "   per-tumour Dice, detection,",
        "   false positives / patient",
        "Must beat best baseline AND",
        "   the control, Wilcoxon p<0.05",
        "   + bootstrap 95% CI",
        "Final: 106 tumours (58 <10 mm)"]),
]
cw, gap, x0 = 18.6, 0.95, 1.5
for i, (h, col, lines) in enumerate(cards):
    x = x0 + i * (cw + gap)
    box(x, 1.2, cw, 21.6, LIGHT, r=0.6, z=1)
    box(x, 19.4, cw, 3.4, col, r=0.6, z=2)
    box(x, 19.4, cw, 1.0, col, r=0.0, z=2)
    text(x + 0.9, 21.1, h, 12.5, "white", "bold", ha="left")
    for j, s in enumerate(lines):
        text(x + 0.9, 17.6 - j * 2.0, s, 10.2, INK, ha="left",
             family="DejaVu Sans Mono" if i == 1 and j < 2 else None)

out = sys.argv[1] if len(sys.argv) > 1 else "final_architecture.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
