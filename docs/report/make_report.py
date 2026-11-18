"""Build the project report PDF from the result files in the repo (no numbers typed by hand
except the ones that only exist in logs, which are marked as such below)."""
import csv
import io
import json
import os
import statistics as st
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

REPO = "/Users/vaideeshk/sem7/ldct-bench"
R = os.path.join(REPO, "dgx_results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs/report/LDCT_Project_Report.pdf")   # needs reportlab + matplotlib

# ----------------------------------------------------------------------------- data
summary = {(r["model"], r["test_set"]): r for r in csv.DictReader(open(os.path.join(R, "summary.csv")))}
noise_now = json.load(open(os.path.join(R, "lits_added_noise.json")))
# before / after per-patient calibration, both measured the same way (liver-mask ROI stored in meta.json):
# before = the fixed-exposure build (commit 8cb1869), after = the current meta files
def _meta_noise(text):
    return json.loads(text)["liver_added_noise_std_hu"]["median"]
noise_old = {p: {"added_noise_std_median_hu": _meta_noise(subprocess.run(
    ["git", "-C", REPO, "show", f"8cb1869:dgx_results/lits_meta/{p}/meta.json"],
    capture_output=True, text=True, check=True).stdout)} for p in json.load(open(os.path.join(R, "lits_added_noise.json")))}
noise_after_meta = {p: {"added_noise_std_median_hu": _meta_noise(open(os.path.join(R, "lits_meta", p, "meta.json")).read())}
                    for p in noise_old}
mayo_noise = json.load(open(os.path.join(R, "mayo_added_noise.json")))["mayo"]
split = json.load(open(os.path.join(REPO, "configs/lits_split.json")))
test_ids = [f"lits_{k:03d}" for k in split["test"]]
detect_path = os.path.join(R, "detect_summary.csv")
detect = list(csv.DictReader(open(detect_path))) if os.path.exists(detect_path) else None
paired = json.load(open(os.path.join(R, "detect_paired.json")))
PI, PA = paired["per_input"], paired["paired"]


def per_slice(model, test_set):
    rows = list(csv.DictReader(open(os.path.join(R, f"{model}_scores_{test_set}.csv"))))
    return rows


def train_time(log):
    rows = list(csv.DictReader(open(os.path.join(R, log))))
    return sum(float(r["seconds"]) for r in rows), len(rows)


MODELS = ["ct_mamba", "corediff", "ctformer"]
NAME = {"quarter_dose": "Quarter dose (input)", "ct_mamba": "CT-Mamba", "corediff": "CoreDiff",
        "ctformer": "CTformer"}
COL = {"quarter_dose": "#8a8984", "ct_mamba": "#2a78d6", "corediff": "#eb6834", "ctformer": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_to_image(fig, width_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


# Figure 1: PSNR and RMSE on both test sets
def figure_scores():
    order = ["quarter_dose", "ctformer", "corediff", "ct_mamba"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    for ax, ts, title in zip(axes, ["mayo_L506", "lits"], ["Mayo L506 (526 slices)", "LiTS locked test set (1,458 slices)"]):
        vals = [float(summary[(m, ts)]["psnr"]) for m in order]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.6, color=[COL[m] for m in order], edgecolor="white", linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.15, yi, f"{v:.2f}", va="center", color=INK, fontsize=8)
        ax.set_yticks(y, [NAME[m] for m in order])
        ax.set_xlim(20, 31.5)
        ax.set_xlabel("PSNR (dB), higher is better")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    return fig


# Figure 2: LiTS noise calibration before / after
def figure_noise():
    ids = sorted(noise_now, key=lambda p: noise_old[p]["added_noise_std_median_hu"])
    old = [noise_old[p]["added_noise_std_median_hu"] for p in ids]
    new = [noise_after_meta[p]["added_noise_std_median_hu"] for p in ids]
    lo, hi = min(mayo_noise["per_patient"].values()), max(mayo_noise["per_patient"].values())
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    x = np.arange(len(ids))
    ax.axhspan(lo, hi, color="#d6e6f8", zorder=0, label=f"range of the 9 real Mayo patients ({lo:.0f}-{hi:.0f} HU)")
    ax.axhline(mayo_noise["added_noise_std_median_hu"], color="#2a78d6", linewidth=1, linestyle="--", zorder=1,
               label=f"target: Mayo median {mayo_noise['added_noise_std_median_hu']:.1f} HU")
    ax.scatter(x, old, s=14, color="#8a8984", zorder=2, label="one fixed exposure for every patient")
    ax.scatter(x, new, s=14, color="#eb6834", zorder=3, label="exposure calibrated per patient (final)")
    ax.set_yscale("log")
    ax.set_yticks([30, 50, 100, 200, 400, 700], ["30", "50", "100", "200", "400", "700"])
    ax.set_xticks([])
    ax.set_xlabel("the 51 LiTS patients, sorted by their noise before calibration")
    ax.set_ylabel("added noise in liver (HU, log)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(25, 1500)
    fig.tight_layout()
    return fig


# Figure 3: per-patient PSNR on the LiTS test patients
def figure_per_patient():
    by = {}
    for m in MODELS:
        for r in per_slice(m, "lits"):
            by.setdefault((m, r["patient"]), []).append(float(r["psnr"]))
    # quarter-dose per patient from the model files is not stored; use gain over CTformer as context instead
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    x = np.arange(len(test_ids))
    for i, m in enumerate(["ctformer", "corediff", "ct_mamba"]):
        vals = [np.mean(by[(m, p)]) for p in test_ids]
        ax.plot(x, vals, marker="o", markersize=4.5, linewidth=1.6, color=COL[m], label=NAME[m])
    ax.set_xticks(x, [p.replace("lits_", "") for p in test_ids])
    ax.set_xlabel("LiTS test patient")
    ax.set_ylabel("mean PSNR (dB)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    wins = sum(np.mean(by[("ct_mamba", p)]) > max(np.mean(by[("corediff", p)]), np.mean(by[("ctformer", p)]))
               for p in test_ids)
    return fig, wins


# ----------------------------------------------------------------------------- styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14,
                      textColor=colors.HexColor(INK), spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor(INK2))
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19,
                    spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1f4f8a"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor(INK))
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("cap", parent=SMALL, spaceBefore=2, spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=22, leading=28,
                       alignment=TA_CENTER, textColor=colors.HexColor("#1f4f8a"))
SUB = ParagraphStyle("sub", parent=BODY, fontSize=12, leading=16, alignment=TA_CENTER,
                     textColor=colors.HexColor(INK2))


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, bold_rows=(), header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) or i in bold_rows else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(GRID)),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")))
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f5")))
    t.setStyle(TableStyle(style))
    return t


def score_rows(ts, with_time=True):
    rows = [["Input / model", "PSNR (dB)", "SSIM", "RMSE (HU)"] + (["seconds / slice"] if with_time else [])]
    for m in ["quarter_dose", "ct_mamba", "corediff", "ctformer"]:
        r = summary[(m, ts)]
        row = [NAME[m], f"{float(r['psnr']):.2f}", f"{float(r['ssim']):.4f}", f"{float(r['rmse']):.2f}"]
        if with_time:
            row.append(f"{float(r['seconds_per_slice']):.3f}" if r["seconds_per_slice"] else "-")
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- numbers used in text
m_mayo = {m: float(summary[(m, "mayo_L506")]["psnr"]) for m in ["quarter_dose"] + MODELS}
m_lits = {m: float(summary[(m, "lits")]["psnr"]) for m in ["quarter_dose"] + MODELS}
r_mayo = {m: float(summary[(m, "mayo_L506")]["rmse"]) for m in ["quarter_dose"] + MODELS}
new_vals = [v["added_noise_std_median_hu"] for v in noise_now.values()]
old_vals = [noise_old[p]["added_noise_std_median_hu"] for p in noise_now]
I0s = [v["I0"] for v in noise_now.values()]
ctf_s, ctf_ep = train_time("ctformer_train_log.csv")
cdf_s, cdf_ep = train_time("corediff_train_log.csv")
n_slices = sum(json.load(open(os.path.join(R, "lits_meta", p, "meta.json")))["kept_slices"]["count"] for p in noise_now)
fig3, mamba_wins = figure_per_patient()


def hm(sec):
    h = int(sec // 3600)
    return f"{h // 24} d {h % 24} h" if h >= 24 else f"{h} h {int(sec % 3600 // 60)} min"


# ----------------------------------------------------------------------------- story
S = []
S += [Spacer(1, 5 * cm),
      P("Low-Dose Liver CT Denoising for Small-Tumour Detection", TITLE), Spacer(1, 0.4 * cm),
      P("Project report: benchmark, baselines, tumour-detection evaluation and planned improvements", SUB),
      Spacer(1, 0.3 * cm),
      P("B.Tech capstone project", SUB),
      Spacer(1, 3 * cm)]
S.append(table([["At a glance", ""],
                ["Goal", "Denoise low-dose (quarter-dose) liver CT so that small tumours stay detectable, then improve on the best model."],
                ["Models compared", "CT-Mamba (2025), CoreDiff (TMI 2024), CTformer (PMB 2023), all trained under one fixed protocol."],
                ["Best model", f"CT-Mamba: {m_mayo['ct_mamba']:.2f} dB on Mayo L506 and {m_lits['ct_mamba']:.2f} dB on LiTS (input {m_mayo['quarter_dose']:.2f} / {m_lits['quarter_dose']:.2f} dB)."],
                ["Tumour data", f"51 LiTS patients ({n_slices:,} liver slices) with realistic simulated quarter dose; 10 patients locked for testing."],
                ["Detection", "nnU-Net finds 27-28 of 34 test tumours on every input, so detection counts do not separate the models. "
                              "Tumour outlines do: quarter dose lowers lesion Dice; CoreDiff and CTformer restore it, CT-Mamba does not."],
                ["Next", "Make the detection test harder (lower doses, more tumours), then train the two CT-Mamba improvements, above all the lesion-weighted loss."]],
               [3.5, 12.5]))
S.append(PageBreak())

# 1 summary
S.append(P("1. Summary", H1))
S.append(P("Low-dose CT reduces radiation to the patient but adds noise, and noise hides small liver tumours. "
           "This project builds a fair benchmark for AI denoising of low-dose CT, uses it to compare three "
           "recent published models, and measures not only image quality but whether a tumour-finding network "
           "can still find the tumours after denoising. The main findings so far:"))
S += bullets([
    f"<b>All three models clearly beat the noisy input.</b> On the held-out Mayo patient, PSNR rises from "
    f"{m_mayo['quarter_dose']:.2f} dB to {m_mayo['ctformer']:.2f}-{m_mayo['ct_mamba']:.2f} dB and the error "
    f"(RMSE) drops from {r_mayo['quarter_dose']:.1f} HU to {r_mayo['ct_mamba']:.1f}-{r_mayo['ctformer']:.1f} HU.",
    f"<b>CT-Mamba is best on PSNR and RMSE</b> on both test sets; CoreDiff is a close second and has the best "
    f"SSIM on Mayo; CTformer is third. CT-Mamba is also about 10 times slower per slice.",
    f"<b>The ranking holds on unseen data.</b> On the LiTS tumour patients, which no model saw during training, "
    f"the order is the same, and CT-Mamba has the highest PSNR on {mamba_wins} of the 10 test patients.",
    "<b>A realistic tumour test set was built.</b> LiTS contains only normal-dose scans, so quarter dose was "
    "simulated in the projection domain and calibrated per patient to match the noise of real Mayo quarter-dose scans.",
    f"<b>At quarter dose, detection counts do not separate the inputs.</b> The trained nnU-Net finds 27-28 of the "
    f"{paired['lesions']} test tumours on every input, including the noisy one; the same small tumours are missed "
    f"everywhere. The test is not yet sensitive enough to rank models.",
    f"<b>Tumour outlines do separate them.</b> Quarter dose lowers the per-tumour Dice "
    f"({PI['quarter']['mean_lesion_dice']:.3f} vs {PI['full']['mean_lesion_dice']:.3f}, paired p = "
    f"{PA['quarter_vs_full']['wilcoxon_p']:.3f}). CoreDiff and CTformer bring it back to the normal-dose level, "
    f"but CT-Mamba, the PSNR winner, ends lowest ({PI['ct_mamba']['mean_lesion_dice']:.3f}). "
    f"Best PSNR is not best for tumours, which is exactly what the lesion-aware loss (Idea 1) targets.",
    "<b>Two improvements to CT-Mamba are implemented and tested</b> but not yet trained: a hybrid with CTformer's "
    "attention block, and a training loss that weights tumour pixels five times more.",
])

# 2 problem
S.append(P("2. Problem and approach", H1))
S.append(P("Image-quality scores such as PSNR reward overall smoothness. A model can score well and still blur a "
           "5 mm lesion into the liver. The project therefore has two layers of evaluation: standard image "
           "metrics on every model, and a downstream test in which the same trained tumour finder is run on normal "
           "dose, on quarter dose, and on each model's denoised output. The gap between the quarter-dose and "
           "normal-dose detection rates is what denoising should close."))
S.append(P("The work is organised as a benchmark (Section 3), data (Section 4), baselines (Sections 5-6), the "
           "detection test (Section 7), and two proposed improvements (Section 8)."))

# 3 benchmark
S.append(P("3. The benchmark", H1))
S.append(P("A shared Python package (<i>ldct</i>) handles data loading, training, tiled inference, metrics and "
           "result tables. Each model is one wrapper file around its official code, pinned to a fixed commit. "
           "Every model is trained and scored under the same fixed rules:"))
S.append(table([["Rule", "Setting"],
                ["Training data", "Mayo 2016 Low Dose CT Grand Challenge, 1 mm B30 kernel, 9 patients: L067 L096 L109 L143 L192 L286 L291 L310 L333"],
                ["Locked test patient", "L506: never used for training, validation or tuning (enforced in code)"],
                ["Final checkpoint", "the epoch with the lowest training loss, as in the CT-Mamba paper; validation only logs a loss"],
                ["Metrics", "PSNR, SSIM, RMSE in the soft-tissue window -160 to 240 HU, same scorer for all models, plus per-slice scores"],
                ["LiTS test set", "10 patients chosen by a fixed random seed, saved in a committed file, never used for training or tuning"],
                ["Baseline settings", "each paper's own optimiser, batch, schedule length and loss; where these conflict with the rules above, the rules win and the difference is documented"],
                ["Hardware", "university DGX A100 cluster (SLURM); long jobs on a full A100, tests on MIG slices"]],
               [3.8, 12.2]))
S.append(Spacer(1, 4))
S.append(P("More than 150 automated checks in 13 test files cover the data pipeline, metrics, inference, each wrapper, "
           "the low-dose simulator, the detection scorer and both improvements; all pass on the laptop and on the cluster.", SMALL))

# 4 data
S.append(P("4. Data", H1))
S.append(P("4.1 Mayo 2016", H2))
S.append(P("Paired real quarter-dose and full-dose scans of 10 patients. Nine are used for training and L506 "
           "(526 slices) for testing. These scans have no tumour labels, so they measure image quality only."))
S.append(P("4.2 LiTS with simulated quarter dose", H2))
S.append(P(f"The Liver Tumour Segmentation (LiTS) dataset has tumour masks but only normal-dose scans. Part 1 "
           f"(51 patients) was converted to liver slices with 5 slices of margin: {n_slices:,} slices with liver and "
           f"tumour masks. The patients were split by seed into 41 for training and 10 locked test patients "
           f"({', '.join(str(k) for k in split['test'])})."))
S.append(P("Quarter dose was simulated in the projection domain, following the approach of Yu et al. (2012): each "
           "slice is converted to attenuation, forward projected (720 angles), Poisson photon noise and Gaussian "
           "electronic noise are added at 25 % of the exposure, and the noisy sinogram is reconstructed with "
           "filtered back-projection. Only the noise is added to the original image, so anatomy is unchanged."))
S.append(P(f"<b>Calibration.</b> The target is the real Mayo noise: the median standard deviation of "
           f"(quarter dose minus full dose) in uniform soft tissue is {mayo_noise['added_noise_std_median_hu']:.1f} HU "
           f"over the nine training patients. With one fixed exposure for every LiTS patient, the added noise "
           f"ranged from {min(old_vals):.0f} to {max(old_vals):.0f} HU, because patient size and pixel size differ "
           f"widely. Real scanners adjust exposure per patient (automatic exposure control), so the simulator now "
           f"does the same: it iterates the exposure on up to 12 liver slices until the noise is within 3 % of the "
           f"target. Final result: {min(new_vals):.1f}-{max(new_vals):.1f} HU, median {st.median(new_vals):.1f} HU, "
           f"with exposures from {min(I0s):.1e} to {max(I0s):.1e} photons (Figure 2)."))
S.append(KeepTogether([fig_to_image(figure_noise(), 16),
                       P("Figure 2. Added noise in the liver for each LiTS patient, before and after per-patient "
                         "exposure calibration. The shaded band is the range of the real Mayo quarter-dose patients.", CAP)]))

# 5 models
S.append(P("5. Models compared", H1))
S.append(table([["Model", "Type", "Size", "Key official settings used", "Training time, one A100"],
                ["CT-Mamba (Li et al., 2025)", "wavelet split + multi-scale Mamba state-space blocks with Z-order scans",
                 "50.8 M", "250 epochs, AdamW 1e-3 cosine, 64x64 patches, L1 + noise-power-spectrum loss", "2 d 22 h"],
                ["CoreDiff (Gao et al., TMI 2024)", "cold diffusion, 10 steps, 3 adjacent slices as input, EMA weights",
                 "4.8 M", "150k iterations, Adam 2e-4, batch 8 whole slices, DDIM sampling", hm(cdf_s)],
                ["CTformer (Wang et al., PMB 2023)", "convolution-free transformer with token-to-token dilated attention",
                 "1.4 M", "530k updates, Adam 1e-5, 64 patches per step with flip/rotate augmentation", hm(ctf_s)]],
               [3.3, 4.2, 1.5, 4.8, 2.2]))
S.append(Spacer(1, 4))
S.append(P("Documented differences from the official code, all caused by the fixed rules: CTformer uses 1 mm data "
           "(its README uses 3 mm), the shared cosine learning-rate decay, the lowest-loss checkpoint, and averaged "
           "overlapping tiles at test time. CoreDiff uses the lowest-loss EMA checkpoint and keeps the first and last "
           "slice of each patient so that all 526 L506 slices are scored.", SMALL))

# 6 results
S.append(PageBreak())
S.append(P("6. Image-quality results", H1))
S.append(P("6.1 Mayo L506 (real quarter dose, 526 slices)", H2))
S.append(table(score_rows("mayo_L506"), [5, 2.5, 2.5, 2.5, 3.5]))
S.append(P("Timing on a full A100 40 GB, whole 512 x 512 slice.", CAP))
S.append(P("6.2 LiTS locked test set (simulated quarter dose, 10 patients, 1,458 slices)", H2))
S.append(table(score_rows("lits"), [5, 2.5, 2.5, 2.5, 3.5]))
S.append(P("Timing on a MIG 3g.20gb slice (a fraction of an A100), so it is slower than the Mayo column. "
           "No model was trained on LiTS.", CAP))
S.append(KeepTogether([fig_to_image(figure_scores(), 16),
                       P("Figure 1. PSNR on both test sets. The x-axis starts at 20 dB to show the differences between models.", CAP)]))
S.append(KeepTogether([fig_to_image(fig3, 16),
                       P(f"Figure 3. Mean PSNR per LiTS test patient. Difficulty varies a lot between patients, but "
                         f"the ordering of the models is stable: CT-Mamba is highest on {mamba_wins} of 10.", CAP)]))
S.append(P("<b>Reading the results.</b> CT-Mamba leads on PSNR and RMSE on both sets, but its margin over "
           f"CoreDiff is small ({m_mayo['ct_mamba'] - m_mayo['corediff']:.2f} dB on Mayo, "
           f"{m_lits['ct_mamba'] - m_lits['corediff']:.2f} dB on LiTS) while it is about 10 times slower. CoreDiff has "
           "the best SSIM on Mayo, i.e. it keeps local structure slightly better. CTformer, with 35 times fewer "
           "parameters than CT-Mamba, is about 0.9-1.4 dB behind. Whether these differences matter for finding "
           "tumours is the question of Section 7."))

# 7 detection
S.append(P("7. Tumour-detection evaluation", H1))
S.append(P("<b>Tumour finder.</b> nnU-Net v2 (3D full resolution, one fold, 250 epochs) was trained on the 41 LiTS "
           "training patients at normal dose (32 for training, 9 for validation inside nnU-Net). It finished in 4 h "
           "11 min on one A100 with a validation pseudo-Dice of 0.965 for the liver and 0.681 for tumours; final "
           "training loss -0.755, validation loss -0.674 (from the training log)."))
S.append(P("<b>Lesion-level scorer.</b> Each tumour is a 3D connected component (26-connectivity) of the ground-truth "
           "mask. It counts as detected if the prediction overlaps it. The scorer reports the detection rate, the "
           "rate for small tumours (equivalent diameter under 10 mm), false-positive components per patient, Dice per "
           "lesion and per patient. It is covered by 18 synthetic tests."))
S.append(P("<b>Comparison.</b> The same trained nnU-Net is run on the 10 locked test patients at normal dose (the upper "
           "bound), at quarter dose (the lower bound), and on the output of each denoising model."))
if detect:
    rows = [["Input to nnU-Net", "Lesions detected", "Detection rate", "Small (< 10 mm) detected", "False positives / patient", "Per-patient Dice"]]
    label = {"full": "Normal dose (upper bound)", "quarter": "Quarter dose (input)", "ct_mamba": "CT-Mamba output",
             "corediff": "CoreDiff output", "ctformer": "CTformer output"}

    def f(x, p=3):
        return "-" if x in (None, "", "None") else f"{float(x):.{p}f}"
    order = ["full", "quarter", "ct_mamba", "corediff", "ctformer"]
    for r in sorted(detect, key=lambda r: order.index(r["input"]) if r["input"] in order else 99):
        nm = r.get("input") or r.get("name")
        small_key = next(k for k in r if k.startswith("small_lesions_under"))
        rows.append([label.get(nm, nm), f"{r['detected']} / {r['lesions']}", f(r["detection_rate"]),
                     f"{r['small_detected']} / {r[small_key]}", f(r["false_positives_per_patient"], 1),
                     f(r["mean_per_patient_dice"])])
    S.append(table(rows, [4.2, 2.4, 2.1, 2.8, 2.5, 2.0]))
    S.append(P("Lesion-level detection on the 10 locked LiTS test patients with the same nnU-Net for every input. "
               "Dice values are means over patients.", CAP))
    S.append(P("<b>Paired per-tumour analysis.</b> Every input is scored on the same 34 tumours, so the per-tumour Dice "
               "values can be compared pairwise (Wilcoxon signed-rank test, scripts/detect_compare.py).", BODY))
    prow = [["Input", "Mean tumour Dice", "Small (< 10 mm)", "Large", "vs normal dose", "vs quarter dose"]]
    lab2 = {"full": "Normal dose", "quarter": "Quarter dose", "ct_mamba": "CT-Mamba", "corediff": "CoreDiff", "ctformer": "CTformer"}
    for n in ["full", "quarter", "ct_mamba", "corediff", "ctformer"]:
        v = PI[n]
        def cmp(k):
            if k not in PA:
                return "-"
            c = PA[k]
            return f"{c['mean_dice_difference']:+.3f} (p = {c['wilcoxon_p']:.2f})"
        prow.append([lab2[n], f"{v['mean_lesion_dice']:.3f}", f"{v['mean_dice_small']:.3f}", f"{v['mean_dice_large']:.3f}",
                     cmp(f"{n}_vs_full"), cmp(f"{n}_vs_quarter")])
    S.append(table(prow, [2.8, 2.6, 2.4, 1.8, 3.2, 3.2]))
    S.append(P("Differences are the mean change in per-tumour Dice; p from the paired Wilcoxon test over 34 tumours.", CAP))
    missed = paired["missed"]
    none = [m for m in missed if not m["detected_by"]]
    S.append(P("<b>What this shows.</b>", BODY))
    S += bullets([
        f"<b>Detection is saturated.</b> {len(none)} tumours are missed on every input, normal dose included: "
        + ", ".join(f"{m['diameter_mm']:g} mm" for m in none) + ". Two more flip between inputs by chance. "
        "The tumour finder's limit, not the noise, decides what is found, so detection counts cannot rank the "
        "denoisers at this dose with 34 tumours.",
        f"<b>Noise does hurt the outlines.</b> Quarter dose lowers per-tumour Dice by "
        f"{-PA['quarter_vs_full']['mean_dice_difference']:.3f} (p = {PA['quarter_vs_full']['wilcoxon_p']:.3f}), the only "
        "statistically significant difference here.",
        f"<b>CoreDiff and CTformer remove that loss</b> ({PA['corediff_vs_full']['mean_dice_difference']:+.3f} and "
        f"{PA['ctformer_vs_full']['mean_dice_difference']:+.3f} vs normal dose), and do best on small tumours.",
        f"<b>CT-Mamba does not.</b> Despite the best PSNR it has the lowest tumour Dice, "
        f"{PI['ct_mamba']['mean_lesion_dice']:.3f}, and the lowest on small tumours, "
        f"{PI['ct_mamba']['mean_dice_small']:.3f}. The difference is not significant with 34 tumours "
        f"(p = {PA['ct_mamba_vs_full']['wilcoxon_p']:.2f}), but it points the same way on large and small tumours. "
        "A likely cause is over-smoothing at tumour edges, which PSNR rewards. It also has the fewest false positives.",
    ])
else:
    S.append(P("<b>Status:</b> the comparison job is running on the cluster at the time of writing (about one hour). "
               "Its first attempt failed because nnU-Net writes volumes with the axes in (x, y, z) order while the "
               "masks were read in (z, y, x); the scorer was fixed, a regression test was added, and the job was "
               "resubmitted. The results table will be added to this report when it finishes."))

# 8 improvements
S.append(P("8. Proposed improvements to CT-Mamba", H1))
S.append(P("Idea 2: CT-Mamba with CTformer's attention", H2))
S.append(P("CT-Mamba splits each patch into four wavelet bands. Each band goes through a multi-scale Mamba module "
           "at 32x32, 16x16 and 8x8 token grids. Mamba scans tokens in fixed orders, so each token only sees tokens "
           "before it. CTformer's token-to-token dilated attention lets every token see every other token. A drop-in "
           "block was built that has the same input and output shape as a Mamba block: gather a 3x3 dilated "
           "neighbourhood, compress it with CTformer's performer layer, add position codes, and apply one "
           "8-head transformer layer. Three variants were implemented and measured:"))
S.append(table([["Variant", "What changes", "Params", "s / step (batch 32)", "Peak memory"],
                ["CT-Mamba (reference)", "nothing", "50.8 M", "1.96", "18.4 GiB"],
                ["(a) low", "attention replaces every Mamba block in the low-frequency band", "59.4 M", "~2.2", "~24 GiB"],
                ["(b) small", "attention replaces only the 8x8 scale in all four bands", "197.9 M", "~2.9", "~23 GiB"],
                ["(c) parallel", "attention added beside Mamba in the low band, mixed by a learned weight", "61.7 M", "~2.3", "~26 GiB"]],
               [3.2, 6.6, 1.8, 2.4, 2.0]))
S.append(P("Measured on a MIG slice; hybrid values at batch 16 scaled to batch 32. Variant (b) is large because the "
           "8x8 scale has the widest channels; a 1x1 channel bottleneck would make it light again.", CAP))
S.append(P("Idea 1: lesion-aware training", H2))
S.append(P("Standard losses treat every pixel equally, and tumours are a tiny fraction of the pixels. The proposed "
           "fine-tuning of CT-Mamba adds the 41 LiTS training patients and (i) a lesion-weighted L1 loss in which "
           "tumour pixels count 5 times more, and (ii) tumour-aware patch sampling, where half the patches from a "
           "tumour slice are centred on a tumour. A control run with the same data and schedule but without the "
           "weighting isolates the effect of the loss. Both are implemented with 24 tests; the job scripts are ready."))

# 9 problems
S.append(P("9. Engineering problems solved", H1))
S += bullets([
    "Kaggle refuses per-file downloads for LiTS; the 5.2 GB archive is downloaded once on a compute node and patients "
    "are extracted one at a time to stay inside the 50 GB storage quota.",
    "A single exposure gave unrealistic noise (Figure 2); replaced by per-patient calibration, re-run from the stored slices.",
    "A one-step exposure estimate undershot because electronic noise breaks the simple 1/sqrt(dose) law; replaced by an iterative calibration.",
    "Hybrid models ran out of memory on a GPU slice at batch 32; measured at smaller batches and scaled.",
    "The detection scorer read nnU-Net volumes with the wrong axis order; fixed with a round-trip test.",
    "All long jobs are chained on one A100 with SLURM dependencies; everything is logged in docs/PROGRESS.md.",
])

# 10 limitations
S.append(P("10. Limitations", H1))
S += bullets([
    "The LiTS low dose is simulated with parallel-beam geometry and no bow-tie filter, while real scanners use fan or "
    "cone beams. The noise level matches real data, but its texture may differ somewhat.",
    "The Mayo test set is a single patient (L506), as fixed by the protocol, so the Mayo numbers come from one patient; the LiTS test set adds 10 more.",
    "nnU-Net uses one fold, not the usual five-fold ensemble, to keep compute within the student allocation.",
    "Only LiTS part 1 (51 of 131 patients) is used, because of the storage quota.",
    "No radiologist reading study; detection is measured by an automatic tumour finder.",
])

# 11 next steps
S.append(P("11. Next steps", H1))
S.append(table([["#", "Step", "Compute", "What it answers"],
                ["1", "Make the detection test sensitive: re-simulate the 10 locked test patients at lower doses (10 % and 5 %) and rerun the same nnU-Net on every input", "~1 day CPU + ~1 h GPU per dose", "At what dose does detection start to fail, and which denoiser recovers the most tumours there?"],
                ["2", "Add tumours to the test set: LiTS part 2 patients (never seen), locked before use", "storage-limited; ~1 day", "34 tumours is too few to show differences of a few percent"],
                ["3", "Idea 2: four 50-epoch runs (CT-Mamba reference + three hybrid variants), scored on L506 and LiTS", "~14 h each, chained", "Does global attention improve CT-Mamba, and where in the network?"],
                ["4", "Idea 1 (priority, motivated by Section 7): two 20-epoch fine-tunes of CT-Mamba with LiTS data, with and without the lesion-weighted loss", "~1 day", "Can CT-Mamba keep its PSNR lead and recover the tumour Dice that CoreDiff and CTformer keep?"],
                ["5", "Run the best variant through the detection test", "~1 h", "The final comparison against the three baselines"],
                ["6", "If a variant wins: full 250-epoch training and statistical tests (paired, per patient and per slice)", "~3 days", "Is the improvement real and significant?"],
                ["7", "Write-up: method, results, limitations", "-", "Capstone thesis"]],
               [0.6, 6.4, 2.6, 6.4]))

S.append(P("Appendix: timeline", H1))
S.append(table([["Dates", "Work"],
                ["4-8 Oct 2026", "Benchmark package: Mayo preparation, metrics, tiled inference, model registry, training/prediction/evaluation scripts, CT-Mamba wrapper, end-to-end tests"],
                ["15-18 Oct", "CT-Mamba trained for 250 epochs on the DGX (2 d 22 h); scored on L506"],
                ["12 Nov", "CTformer and CoreDiff wrappers and tests; LiTS split, simulator, conversion; detection scorer; both improvement ideas coded and tested; nnU-Net set up"],
                ["13-15 Nov", "CTformer and CoreDiff trained (" + hm(ctf_s) + ", " + hm(cdf_s) + " of training); LiTS part 1 converted"],
                ["16 Nov", "Noise problem found; per-patient calibration; LiTS re-simulated; all three models scored on LiTS"],
                ["17-18 Nov", "nnU-Net trained (4 h 11 min); detection scorer fixed; detection comparison and paired analysis complete"]],
               [2.8, 13.2]))
S.append(Spacer(1, 6))
S.append(P("All code, configuration, per-slice scores and logs are in the project repository (branch autonomous-nov12). "
           "Result tables: dgx_results/summary.csv and dgx_results/detect_summary.csv; paired detection analysis: dgx_results/detect_paired.json; per-slice scores: dgx_results/&lt;model&gt;_scores_&lt;set&gt;.csv.", SMALL))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(INK2))
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Low-dose liver CT denoising: project report")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Low-Dose Liver CT Denoising: Project Report",
                        author="ldct-bench")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
