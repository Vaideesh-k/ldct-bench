const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/vaideeshk/.claude/skills/synced/16136379-3480-4298-a3d0-134bd2292863_d0b2a7f4-3936-4224-8c21-e87c4c8fe596/pptx/scripts/apply_theme.js");

const HEX = {
  dk1: "14232E", lt1: "FFFFFF", dk2: "0F2A3D", lt2: "EEF3F5",
  accent1: "0E8A96", accent2: "E3893B", accent3: "5B6F7C", accent4: "9CC9CF",
  accent5: "B8443A", accent6: "3E7C4A", hlink: "0E8A96", folHlink: "5B6F7C",
};
const THEME = { name: "LDCT Liver", headFontFace: "Cambria", bodyFontFace: "Calibri", colors: HEX };

const PLAN = process.env.PLAN === "1";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Low-dose CT denoising for small liver tumours";
pres.author = "B.Tech capstone team";
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
const C = pres.SchemeColor;
const W = 13.333;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", fontFace: "Cambria" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 11.7, h: 1.6, fontSize: 18, color: C.accent4, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 36, bold: true, color: C.background1, valign: "middle", fontFace: "Cambria" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 1.0, fontSize: 18, color: C.accent4, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent4 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria" }, text: "" } },
    { text: { text: "LDCT denoising for small liver tumours  |  progress review, 19 Nov 2026", options: { x: 0.6, y: 6.95, w: 8, h: 0.3, fontSize: 10, color: HEX.accent3 } } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent3 },
});

// ---------------------------------------------------------------- helpers
let n = 0;
function content(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function card(s, x, y, w, h, head, body, opt = {}) {
  n++;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opt.fill || C.background2 }, line: { color: opt.fill || C.background2 }, objectName: `card${n}` });
  const runs = [];
  if (head) runs.push({ text: head, options: { bold: true, fontSize: opt.headSize || 16, color: opt.headColor || C.text2, breakLine: true, paraSpaceAfter: 4 } });
  const lines = Array.isArray(body) ? body : [body];
  lines.forEach((t, i) => runs.push({ text: t, options: { fontSize: opt.size || 14, color: opt.color || C.text1, bullet: opt.bullet ? { indent: 14 } : false, breakLine: i < lines.length - 1, paraSpaceAfter: 3 } }));
  s.addText(runs, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2, valign: "top", isTextBox: true, margin: 2, objectName: `cardtext${n}` });
}
function stat(s, x, y, w, num, label, color) {
  n++;
  s.addText(num, { x, y, w, h: 0.85, fontSize: 40, bold: true, color: color || C.accent1, fontFace: "Cambria", isTextBox: true, margin: 0, objectName: `stat${n}` });
  s.addText(label, { x, y: y + 0.85, w, h: 0.6, fontSize: 12, color: C.accent3, isTextBox: true, margin: 0, valign: "top", objectName: `statlab${n}` });
}
function circleNum(s, x, y, d, txt, fill) {
  n++;
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { color: fill || C.accent1 }, objectName: `circ${n}` });
  s.addText(txt, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: `circt${n}` });
}
function note(s, text, y = 6.35) {
  s.addText(text, { x: 0.6, y, w: 12.1, h: 0.45, fontSize: 12, italic: true, color: C.accent3, isTextBox: true, margin: 0, objectName: `note${++n}` });
}
const HDR = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14 } });
function table(s, rows, opt) {
  const body = rows.map((r, i) => i === 0 ? r.map(HDR) : r.map((c) => (typeof c === "object" ? c : { text: String(c), options: { fontSize: 14, color: C.text1, fill: { color: i % 2 ? C.background1 : C.background2 } } })));
  s.addTable(body, Object.assign({ border: { type: "solid", pt: 0.5, color: "D5DEE2" }, fontFace: "Calibri", valign: "middle", margin: 0.06 }, opt));
}
const HL = (t) => ({ text: t, options: { bold: true, fontSize: 14, color: C.accent1, fill: { color: "E2F1F2" } } });
const BAD = (t) => ({ text: t, options: { bold: true, fontSize: 14, color: C.accent5, fill: { color: "F7E6E4" } } });
const chartBase = {
  catAxisLabelColor: HEX.accent3, valAxisLabelColor: HEX.accent3, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, valGridLine: { color: "E1E7EA", size: 0.75 }, catGridLine: { style: "none" },
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, titleFontFace: "+mn-lt", titleColor: HEX.dk2, titleFontSize: 14,
  legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.dk1,
};

// ================================================================= 1 title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Low-dose CT denoising for small liver tumours", { placeholder: "title" });
s.addText("Does a denoiser that wins on image quality also help find tumours under 10 mm?\nB.Tech capstone  |  " + (PLAN ? "progress review, final architecture and training plan" : "progress review and plan for the final week") + "  |  19 November 2026", { placeholder: "body" });
s.addNotes("Introduce the project: we compare deep-learning denoisers for quarter-dose CT, and test them not only on image scores but on whether a tumour segmenter still finds small liver tumours.");

// ================================================================= 2 problem
s = content("Introduction", "Why this project");
card(s, 0.6, 1.45, 5.9, 2.35, "The problem", [
  "Lowering the CT dose to a quarter cuts radiation for the patient, but adds noise.",
  "Noise hides small, low-contrast lesions such as liver tumours under 10 mm.",
], { bullet: true, size: 16 });
card(s, 0.6, 4.0, 5.9, 2.2, "What papers usually report", [
  "Denoisers are ranked on PSNR / SSIM against the normal-dose image.",
  "A higher PSNR does not prove that a tumour is still visible.",
], { bullet: true, size: 16 });
card(s, 6.85, 1.45, 5.85, 4.75, "Our research question", [
  "1. Which of three recent denoisers (CT-Mamba, CTformer, CoreDiff) gives the best image quality on quarter-dose CT?",
  "2. Does the denoised image let a fixed tumour segmenter (nnU-Net) find and outline liver tumours as well as normal dose, especially small ones?",
  "3. Can a small add-on network trained with a tumour-aware loss fix what the best denoiser loses?",
], { fill: C.text2, headColor: C.accent4, color: C.background1, size: 18, headSize: 20 });
s.addNotes("The gap: image metrics are averaged over every pixel; tumours are a tiny fraction of pixels. So we evaluate with a downstream task.");

// ================================================================= 3 pipeline
s = content("Introduction", "End-to-end pipeline");
const steps = [
  ["Data", "Mayo LDCT (real quarter dose) + LiTS liver tumours"],
  ["Low dose", "Quarter dose simulated for LiTS, matched to Mayo noise"],
  ["Denoise", "CT-Mamba, CTformer, CoreDiff trained on Mayo"],
  ["Image scores", "PSNR, SSIM, RMSE per slice on locked test sets"],
  ["Tumour task", "Same nnU-Net on every input; per-tumour detection + Dice"],
  ["Statistics", "Paired Wilcoxon tests, bootstrap 95% CIs"],
];
const bw = 1.85, gap = 0.2;
steps.forEach(([h, b], i) => {
  const x = 0.6 + i * (bw + gap);
  circleNum(s, x + bw / 2 - 0.35, 1.7, 0.7, String(i + 1), i >= 4 ? C.accent2 : C.accent1);
  card(s, x, 2.6, bw, 2.6, h, b, { size: 15, headSize: 17 });
  if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.02, y: 2.0, w: 0.16, h: 0.14, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: `arrow${i}` });
});
card(s, 0.6, 5.45, 12.1, 0.8, null, "Locked test data: Mayo patient L506 and 10 LiTS patients are never used for training, tuning or checkpoint choice.", { size: 16 });
s.addNotes("Steps 1-4 are the classic denoising benchmark. Steps 5-6 are our contribution: a downstream tumour task with paired statistics.");

// ================================================================= section: data + models
pres.addSection({ title: "Data and models" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Data and models" });
s.addText("Data and models", { placeholder: "title" });
s.addText("Two datasets, one noise level, three denoisers", { placeholder: "body" });

// ================================================================= 4 datasets
s = content("Data and models", "Datasets and locked test splits");
stat(s, 0.6, 1.4, 2.8, "5,410", "Mayo training slices (9 patients)");
stat(s, 3.6, 1.4, 2.8, "526", "Mayo test slices (patient L506)");
stat(s, 6.6, 1.4, 2.8, "7,274", "LiTS slices built (51 patients)");
stat(s, 9.6, 1.4, 3.1, "34", "tumours in 10 locked LiTS test patients (9 under 10 mm)", C.accent2);
table(s, [
  ["Dataset", "What it gives us", "Train", "Test (locked)"],
  ["Mayo LDCT challenge", "Real normal- and quarter-dose scans, 1 mm", "L067 L096 L109 L143 L192 L286 L291 L310 L333", "L506, 526 slices"],
  ["LiTS part 1 (vol. 0-50)", "Liver + tumour masks; quarter dose simulated", "41 patients", "2 4 10 11 22 26 28 29 32 45 (1,458 slices)"],
  ["LiTS part 2 (vol. 51-130)", "Extended test set for the final verdict", "never used", "10 new patients, 72 tumours"],
], { x: 0.6, y: 3.15, w: 12.1, colW: [2.6, 3.9, 3.3, 2.3] });
note(s, "The LiTS split (seed 0, 20% test) was committed before any training on LiTS.", 5.6);
s.addNotes("Mayo gives real low-dose scans but no tumour masks. LiTS has tumour masks but no low-dose scans, so we simulate them.");

// ================================================================= 5 noise simulation
s = content("Data and models", "Simulating quarter dose for LiTS");
card(s, 0.6, 1.45, 6.0, 4.8, "Method", [
  "Projection-domain noise: forward projection (Radon), Poisson photon noise + electronic noise, reconstruction (filtered back-projection).",
  "Target: the added noise in Mayo's real quarter-dose liver = 48.2 HU (median of 9 training patients, IQR 41.3-54.6).",
  "First try (one exposure for everyone) failed: added noise ranged 32-686 HU across patients.",
  "Fix: calibrate the photon count PER PATIENT (like a scanner's automatic exposure control), iterating until within 3% of 48.2 HU.",
], { bullet: true, size: 16 });
s.addChart(pres.charts.BAR, [{ name: "Added liver noise (HU)", labels: ["Mayo real (target)", "LiTS fixed exposure, median", "LiTS per-patient, median"], values: [48.2, 70, 48.5] }], Object.assign({}, chartBase, {
  x: 6.9, y: 1.45, w: 5.8, h: 3.6, barDir: "col", chartColors: [HEX.accent3, HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
  showTitle: true, title: "Added noise in the liver (HU)", showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 80, varyColors: true,
}));
card(s, 6.9, 5.25, 5.8, 1.0, null, "Per-patient result: 45.9-51.9 HU over all 51 patients (median 48.5). Quarter-dose LiTS test PSNR 24.29 dB vs Mayo 24.47 dB.", { size: 13 });
note(s, "Limitation: parallel-beam geometry without bow-tie filter; a real scanner is fan/cone beam.", 6.4);
s.addNotes("This calibration makes the LiTS test as hard as the real Mayo quarter dose; without it some patients would have had ten times more noise.");

// ================================================================= 6 models
s = content("Data and models", "The three denoisers");
const models = [
  ["CT-Mamba", "State-space (Mamba) model on wavelet bands", ["50.8 M parameters", "Best checkpoint: epoch 237", "64 x 64 patches", "1.06 s per slice"]],
  ["CTformer", "Convolution-free token-to-token transformer", ["1.45 M parameters", "530,000 steps (12 h 22 min)", "Official settings", "0.11 s per slice"]],
  ["CoreDiff", "Cold-diffusion model with context slices", ["4.83 M parameters (+ EMA copy)", "150,294 steps (13 h 54 min)", "Official settings", "0.09 s per slice"]],
];
models.forEach(([h, sub, pts], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 1.45, 3.9, 3.6, h, [sub, ...pts], { size: 17, headSize: 22, bullet: true });
});
note(s, "All three trained on the same 9 Mayo patients, timed on an NVIDIA A100 40 GB. Official repositories pinned (CTformer 3c62f58, CoreDiff 9bc31dc); every change is documented.", 5.4);
s.addNotes("CTformer and CoreDiff were run with the official schedules; the only differences (1 mm data, cosine schedule, best-by-train-loss) are listed in the code docstrings.");

// ================================================================= section: results
pres.addSection({ title: "Results so far" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Results so far" });
s.addText("Results so far", { placeholder: "title" });
s.addText("Image quality, then the tumour task", { placeholder: "body" });

// ================================================================= 7 image quality chart
s = content("Results so far", "Image quality: CT-Mamba has the best PSNR on both test sets");
s.addChart(pres.charts.BAR, [
  { name: "Mayo L506 (526 slices)", labels: ["Quarter dose", "CTformer", "CoreDiff", "CT-Mamba"], values: [24.47, 28.56, 29.28, 29.42] },
  { name: "LiTS test (1,458 slices)", labels: ["Quarter dose", "CTformer", "CoreDiff", "CT-Mamba"], values: [24.29, 28.51, 29.43, 29.92] },
], Object.assign({}, chartBase, {
  x: 0.6, y: 1.4, w: 7.4, h: 4.9, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent3, HEX.accent1], showValue: true, dataLabelPosition: "outEnd",
  dataLabelFormatCode: "0.00", showTitle: true, title: "PSNR (dB), higher is better", showLegend: true, legendPos: "b", valAxisMinVal: 20, valAxisMaxVal: 32,
}));
stat(s, 8.4, 1.5, 4.3, "+4.95 dB", "CT-Mamba over quarter dose on Mayo L506 (29.42 vs 24.47)");
stat(s, 8.4, 3.05, 4.3, "+5.63 dB", "CT-Mamba over quarter dose on LiTS (29.92 vs 24.29)");
card(s, 8.4, 4.6, 4.3, 1.7, null, "Same ranking on both test sets: CT-Mamba > CoreDiff > CTformer >> quarter dose. CoreDiff has the best SSIM on Mayo.", { size: 14 });
s.addNotes("The ranking holds on simulated LiTS data as well, which suggests our simulation behaves like the real Mayo quarter dose.");

// ================================================================= 8 image quality table
s = content("Results so far", "Image quality in numbers");
table(s, [
  ["Input", "Mayo PSNR (dB)", "Mayo SSIM", "Mayo RMSE (HU)", "LiTS PSNR (dB)", "LiTS SSIM", "LiTS RMSE (HU)", "Time / slice"],
  ["Quarter dose", "24.47", "0.8382", "24.64", "24.29", "0.8193", "25.77", "-"],
  ["CTformer", "28.56", "0.8680", "15.28", "28.51", "0.8354", "15.71", "0.11 s"],
  ["CoreDiff", "29.28", HL("0.8817"), "14.07", "29.43", "0.8560", "14.08", HL("0.09 s")],
  [{ text: "CT-Mamba", options: { bold: true, fontSize: 14, color: C.text1, fill: { color: C.background1 } } }, HL("29.42"), "0.8776", HL("13.85"), HL("29.92"), HL("0.8608"), HL("13.53"), "1.06 s"],
], { x: 0.6, y: 1.5, w: 12.1, colW: [1.9, 1.45, 1.35, 1.55, 1.5, 1.3, 1.55, 1.5] });
card(s, 0.6, 4.4, 5.9, 1.8, "Reading the table", ["Best value per column is highlighted.", "RMSE is in Hounsfield units, so CT-Mamba cuts the error from ~25 HU to ~14 HU."], { bullet: true, size: 16 });
card(s, 6.8, 4.4, 5.9, 1.8, "Cost", ["CT-Mamba is ~10x slower per slice than CoreDiff and CTformer.", "LiTS timings were on a smaller GPU slice (MIG 3g.20gb): CT-Mamba 2.32 s."], { bullet: true, size: 16 });
s.addNotes("Mayo timings on a full A100. Per-slice CSVs for every model are in dgx_results/.");

// ================================================================= 9 detection setup
s = content("Results so far", "Tumour task: one fixed nnU-Net for every input");
card(s, 0.6, 1.45, 5.9, 4.75, "Setup", [
  "nnU-Net v2, 3D full resolution, 250 epochs, fold 0.",
  "Trained ONLY on normal-dose LiTS training patients (32 train / 9 validation).",
  "Validation pseudo-Dice: liver 0.965, tumour 0.681.",
  "At test time it segments normal dose, quarter dose and each denoiser's output of the same 10 patients, so every tumour is scored 5 times (paired).",
], { bullet: true, size: 17 });
card(s, 6.8, 1.45, 5.9, 4.75, "Definitions (3D, 26-connectivity)", [
  "Tumour = one connected component of the ground-truth mask.",
  "Detected = at least one predicted voxel overlaps it.",
  "Small = equivalent sphere diameter < 10 mm.",
  "False positive = predicted component touching no tumour (per patient).",
  "Per-tumour Dice = Dice inside the tumour's box, dilated by 1 voxel.",
], { bullet: true, size: 17 });
s.addNotes("Keeping the segmenter fixed means any change in tumour scores comes from the input image, not from retraining.");

// ================================================================= 10 detection table
s = content("Results so far", "Detection on the 10 locked LiTS patients (34 tumours)");
table(s, [
  ["Input", "Detected", "Small detected (< 10 mm)", "FP / patient", "Per-patient Dice", "Per-tumour Dice", "Small-tumour Dice"],
  ["Normal dose", "27 / 34", "5 / 9", "0.6", "0.605", "0.582", "0.299"],
  ["Quarter dose", "28 / 34", "5 / 9", "0.5", "0.601", "0.565", "0.276"],
  ["CTformer", "28 / 34", "5 / 9", "0.5", "0.599", HL("0.585"), "0.307"],
  ["CoreDiff", "28 / 34", "5 / 9", "0.5", "0.594", "0.584", HL("0.310")],
  ["CT-Mamba", "27 / 34", "5 / 9", HL("0.3"), "0.576", BAD("0.543"), BAD("0.248")],
], { x: 0.6, y: 1.5, w: 12.1, colW: [1.8, 1.4, 2.4, 1.5, 1.7, 1.6, 1.7] });
card(s, 0.6, 4.5, 5.9, 1.7, "Detection counts are saturated", "5 tumours (2.2, 2.7, 5.5, 7.1 and 20.5 mm) are missed on EVERY input, and two more flip by chance. With 34 tumours, counts cannot separate the models.", { size: 14 });
card(s, 6.8, 4.5, 5.9, 1.7, "The outline quality does differ", "CT-Mamba, the PSNR winner, gives the LOWEST per-tumour Dice, also on small tumours, but the fewest false positives.", { size: 14 });
s.addNotes("Highlighted: best per column in teal, worst in red.");

// ================================================================= 11 key finding chart
s = content("Results so far", "Key finding: best PSNR is not the best tumour outline");
s.addChart(pres.charts.BAR, [{ name: "Mean per-tumour Dice difference vs normal dose", labels: ["Quarter dose", "CT-Mamba", "CoreDiff", "CTformer"], values: [-0.017, -0.039, 0.003, 0.003] }], Object.assign({}, chartBase, {
  x: 0.6, y: 1.4, w: 7.2, h: 4.9, barDir: "bar", chartColors: [HEX.accent5, HEX.accent5, HEX.accent1, HEX.accent1], varyColors: true, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;-0.000",
  showTitle: true, title: "Per-tumour Dice minus normal dose (34 paired tumours)", showLegend: false, valAxisMinVal: -0.05, valAxisMaxVal: 0.02, valAxisLabelFormatCode: "0.00",
}));
table(s, [
  ["Comparison", "Diff.", "Better / worse", "Wilcoxon p"],
  ["Quarter vs normal", "-0.017", "4 / 9", BAD("0.018")],
  ["CT-Mamba vs normal", "-0.039", "8 / 13", "0.16"],
  ["CoreDiff vs normal", "+0.003", "10 / 9", "0.93"],
  ["CTformer vs normal", "+0.003", "9 / 7", "0.91"],
  ["CTformer vs quarter", "+0.020", "11 / 4", "0.074"],
], { x: 8.1, y: 1.5, w: 4.6, colW: [1.85, 0.75, 1.1, 0.9] });
card(s, 8.1, 4.55, 4.6, 1.75, null, "Quarter dose measurably hurts tumour outlines (the only significant result). CoreDiff and CTformer recover it; CT-Mamba does not. This motivates the final-week experiment.", { size: 13 });
s.addNotes("Better/worse counts tumours whose Dice changed by more than 0.01. The CT-Mamba drop is larger than the quarter-dose drop but not significant with only 34 tumours: we need more tumours.");

// ================================================================= 12 engineering problems
s = content("Results so far", "Problems we hit and how we solved them");
const probs = [
  ["Kaggle download", "Per-file download of LiTS returns 404; the full archives (17 GB) do not fit the 50 GB quota.", "Read single files straight from the remote ZIP with HTTP range requests (CRC-checked): a 39 MB mask in 25 s, nothing stored."],
  ["Unrealistic noise", "One fixed exposure gave 32-686 HU added noise across patients.", "Per-patient exposure calibration to the Mayo target (45.9-51.9 HU)."],
  ["Axis-order bug", "nnU-Net NIfTI (x, y, z) vs our slices (z, y, x): scoring failed.", "Fixed the loader + a round-trip test; re-ran detection."],
  ["Shared cluster", "Only one GPU job at a time per account; 39 jobs queued for full A100s.", "Single chained job queue on free MIG slices; all jobs resumable."],
];
probs.forEach(([h, p, f], i) => {
  const y = 1.45 + i * 1.22;
  circleNum(s, 0.6, y + 0.2, 0.6, String(i + 1), C.accent2);
  s.addText([{ text: h, options: { bold: true, fontSize: 15, color: C.text2, breakLine: true } }, { text: p, options: { fontSize: 14, color: C.text1 } }], { x: 1.4, y, w: 5.4, h: 1.05, valign: "middle", isTextBox: true, margin: 0, objectName: `prob${i}` });
  card(s, 7.0, y, 5.7, 1.05, null, f, { size: 14 });
});
s.addNotes("Each of these is logged with dates in docs/PROGRESS.md.");

// ================================================================= section: final week
pres.addSection({ title: "Final week" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Final week" });
s.addText("Final week: a tumour-aware refiner", { placeholder: "title" });
s.addText("Can we keep CT-Mamba's image quality and fix its tumour outlines?", { placeholder: "body" });

// ================================================================= 13 refiner design
s = content("Final week", "The refiner: a small U-Net on top of frozen CT-Mamba");
// diagram
const boxes = [
  [0.6, "Quarter-dose\nslice", C.accent3],
  [3.0, "CT-Mamba\n(frozen, Stage A)", C.text2],
  [5.6, "2D U-Net\n[noisy, Stage A]", C.accent1],
  [8.2, "+ residual\n(zero-init)", C.accent2],
  [10.6, "Refined\nslice", C.accent3],
];
boxes.forEach(([x, t, col], i) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 2.1, h: 1.1, rectRadius: 0.1, fill: { color: col }, line: { color: col }, objectName: `dg${i}` });
  s.addText(t, { x, y: 1.55, w: 2.1, h: 1.1, align: "center", valign: "middle", fontSize: 14, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: `dgt${i}` });
  if (i < boxes.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.15, y: 1.95, w: 0.4, h: 0.3, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: `dga${i}` });
});
card(s, 0.6, 3.0, 3.9, 3.2, "Architecture", ["4 levels: 32 / 64 / 128 / 256 channels", "2 x (3x3 conv + GroupNorm + SiLU) per level", "Bilinear upsampling + skip connections", "2,143,169 parameters (4% of CT-Mamba)"], { bullet: true, size: 16 });
card(s, 4.7, 3.0, 3.9, 3.2, "Tumour-aware loss", ["L1 + lesion-weighted L1", "Tumour pixels count 5x in total", "Masks dilated by 3 pixels", "Masks used ONLY in the loss, never at test time"], { bullet: true, size: 16 });
card(s, 8.8, 3.0, 3.9, 3.2, "Training", ["128 x 128 patches, batch 16", "50% Mayo / 50% LiTS train; ~30% of LiTS patches contain tumour", "AdamW, lr 2e-4, cosine, 30,000 iterations", "26 unit checks written first, all pass"], { bullet: true, size: 16 });
s.addNotes("Zero-init means the refiner starts as an exact copy of CT-Mamba and can only move away from it if the loss asks it to.");

// ================================================================= 13b final architecture figure
s = pres.addSlide({ sectionTitle: "Final week" });
s.background = { color: "FFFFFF" };
s.addImage({ path: __dirname + "/final_architecture.png", x: 1.47, y: 0.3, w: 10.4, h: 6.5, altText: "Final architecture: frozen CT-Mamba (Stage A) feeding a 2D U-Net refiner (Stage B, 32/64/128/256 channels, zero-init residual head) whose output is added to the Stage-A image; below, five cards on data, loss, optimisation, the fair control and evaluation.", objectName: "archfig" });
s.addNotes("Walk through left to right: noisy slice, frozen CT-Mamba, concatenate noisy + Stage A, U-Net, zero-init residual added back. Then the five training cards. Stress: masks only in the loss, nothing extra at test time; the control uses the same data, loss and pixel budget.");

// ================================================================= 14 control + criteria
s = content("Final week", "A fair control and success criteria fixed in advance");
card(s, 0.6, 1.45, 4.6, 4.8, "Control experiment", [
  "Fine-tune CT-Mamba itself (from its best checkpoint) with the SAME data, sampling and loss.",
  "Same training pixels: refiner 16 x 128² = control 64 x 64² = 262,144 pixels per step, 30,000 steps (7.9 billion pixels each).",
  "Answers: is any gain from the refiner design, or just from more training on LiTS with this loss?",
], { bullet: true, size: 16 });
table(s, [
  ["", "Criterion (committed before any result)", "Threshold"],
  ["a", "Image quality kept on Mayo L506", "PSNR ≥ 29.42 dB"],
  ["b", "Per-tumour Dice", "≥ best baseline"],
  ["c", "Beats the control", "Dice > control"],
  ["d", "Significant", "Wilcoxon p < 0.05 AND bootstrap 95% CI excludes 0"],
  ["e", "No extra false alarms", "FP / patient ≤ CT-Mamba (0.3)"],
], { x: 5.5, y: 1.5, w: 7.2, colW: [0.45, 3.65, 3.1] });
card(s, 5.5, 4.85, 7.2, 1.4, null, "Bootstrap: 10,000 resamples, seed 0. (b) or (c) without (d) counts as FAIL. The extended 20-patient test set gives the final verdict (docs/success_criteria.md).", { size: 13 });
s.addNotes("Writing the criteria down before training prevents us from choosing a metric after seeing the results.");

// ================================================================= 15 extended test set
s = content("Final week", "Extended test set: three times more tumours");
stat(s, 0.6, 1.4, 2.8, "80", "LiTS part-2 patients scanned (masks only)");
stat(s, 3.6, 1.4, 2.8, "626", "tumours found, 330 under 10 mm");
stat(s, 6.6, 1.4, 2.8, "106", "tumours in the extended test set", C.accent2);
stat(s, 9.6, 1.4, 3.1, "58", "of them under 10 mm (was 9)", C.accent2);
s.addChart(pres.charts.BAR, [
  { name: "Under 10 mm", labels: ["Current test set (10 patients)", "Extended test set (20 patients)"], values: [9, 58] },
  { name: "10 mm or larger", labels: ["Current test set (10 patients)", "Extended test set (20 patients)"], values: [25, 48] },
], Object.assign({}, chartBase, {
  x: 0.6, y: 3.1, w: 6.6, h: 3.2, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.accent2, HEX.accent1], showValue: true, dataLabelPosition: "ctr", dataLabelColor: HEX.lt1,
  showTitle: true, title: "Tumours available for testing", showLegend: true, legendPos: "b", barGapWidthPct: 50,
}));
card(s, 7.5, 3.1, 5.2, 3.2, "Selection rule (committed before scoring)", [
  "New patients 51 58 63 71 79 88 92 112 118 120: 72 tumours, 1,156 slices.",
  "Ordered by number of small tumours; stop at 100+ tumours in total.",
  "Patients with > 10 tumours excluded, so one patient (lits_117, 75 tumours) cannot dominate.",
  "Never used for training or tuning.",
], { bullet: true, size: 14 });
s.addNotes("More small tumours gives the paired tests enough power to detect a Dice difference of the size we saw for CT-Mamba.");

// ================================================================= 16 status / timeline
s = content("Final week", "Status today and schedule");
table(s, [
  ["Step", "Status", "Numbers / ETA"],
  ["1  Success criteria", HL("Done"), "Committed before any result"],
  ["2  Precompute CT-Mamba outputs (Stage A)", BAD("Stopped: disk limit"), "~5 GB int16, ~5 h"],
  ["3  Refiner code + 26 tests", HL("Done"), "2.14 M parameters"],
  ["3  Refiner training", BAD("Blocked: needs step 2"), "30,000 iterations, ~2-4 h"],
  ["4  Control fine-tune", HL("Done"), "30,000 its in ~10.9 h, best window loss 0.004045"],
  ["5  First evaluation (10 patients)", BAD("Control eval stopped: disk"), "Images + detection + statistics, ~20 Nov"],
  ["6  Extended set: scan + split", HL("Done"), "106 tumours, 58 small"],
  ["6  Extended set: conversion + scoring", "Waiting for disk", "~1.5 GB data, ~5 h GPU"],
  ["7  Report tables, figures, curves", "Scripts ready", "After step 6"],
], { x: 0.6, y: 1.45, w: 12.1, colW: [4.6, 2.6, 4.9], rowH: 0.45 });
note(s, "Constraint: the account can run only one GPU job at a time, so all GPU steps run as one chain.", 6.4);
s.addNotes("Deadline: 7 days in total. First evaluation on the existing 10 patients is planned within 3 days.");

// ================================================================= 17 risks
s = content("Final week", "Risks and limitations");
const risks = [
  ["One GPU job at a time", "Refiner and control cannot run in parallel; queue waits for free MIG slices.", "Chained, resumable jobs; shorten both runs equally if needed."],
  ["50 GB storage quota", "Home at ~40 GB, rule keeps 10 GB free; next steps need ~8\u00A0GB.", "Archive rebuildable data to an external laptop, then delete on the cluster."],
  ["Simulated low dose", "LiTS quarter dose is simulated (parallel beam, no bow-tie).", "Calibrated to real Mayo noise; Mayo L506 stays as a real-dose check."],
  ["Small sample", "34 tumours cannot separate models on detection counts.", "Extended set (106 tumours, 58 small) + bootstrap CIs."],
];
risks.forEach(([h, r, m], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 2.45;
  card(s, x, y, 5.95, 2.25, h, [r, "Mitigation: " + m], { size: 16, headSize: 18 });
});
s.addNotes("Also: nnU-Net trained for 250 epochs on one fold instead of the default 1000 x 5 folds, to fit the schedule.");

// ================================================================= TRAINING PLAN (only in the plan deck)
if (PLAN) {
pres.addSection({ title: "Training plan" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Training plan" });
s.addText("Training plan for the new model", { placeholder: "title" });
s.addText("From freeing disk space to the final verdict on 106 tumours", { placeholder: "body" });

// ---- step by step
s = content("Training plan", "Training plan, step by step");
const tsteps = [
  ["Free disk", "Archive rebuildable data to the laptop, then delete it here: frees ~8 GB", "prerequisite"],
  ["Stage A", "Frozen CT-Mamba on 11,226 train slices (Mayo + LiTS), saved as int16", "~6 h, ~5 GB"],
  ["Train refiner", "26 checks + smoke run, then 30,000 iterations (resumable)", "~2-4 h, MIG 4g"],
  ["Evaluate", "Refiner and control on L506 + 10 LiTS patients: images + detection", "~1.5 h each"],
  ["Statistics", "Paired Wilcoxon + bootstrap 95% CIs; pass / fail for criteria (a)-(e)", "minutes, CPU"],
  ["Extended set", "Convert 10 new LiTS patients (CPU), score all 7 inputs on 106 tumours", "~5 h GPU"],
  ["Report", "Tables, zoomed small-tumour figure, training curves", "final day"],
];
tsteps.forEach(([h, b, t], i) => {
  const y = 1.4 + i * 0.73;
  circleNum(s, 0.6, y + 0.04, 0.55, String(i + 1), i === 0 ? C.accent5 : (i < 3 ? C.accent1 : C.accent2));
  s.addText(h, { x: 1.35, y, w: 2.1, h: 0.63, fontSize: 16, bold: true, color: C.text2, valign: "middle", isTextBox: true, margin: 0, objectName: `tsh${i}` });
  card(s, 3.5, y, 7.0, 0.63, null, b, { size: 14 });
  card(s, 10.7, y, 2.0, 0.63, null, t, { size: 14, fill: "E2F1F2" });
});
s.addNotes("Steps 2-4 are GPU work and must run one after the other: the account allows one GPU job at a time. Step 6's conversion is CPU-only and runs in parallel.");

// ---- hyperparameters side by side
s = content("Training plan", "Training settings: refiner vs control");
table(s, [
  ["Setting", "Refiner (new model)", "Control (CT-Mamba fine-tune)"],
  ["What is trained", HL("U-Net only, 2.14 M parameters"), "CT-Mamba denoiser, 50.8 M parameters"],
  ["Starting point", "Zero-init head = exactly CT-Mamba's output", "CT-Mamba best checkpoint (epoch 237)"],
  ["Input", "2 channels: [noisy, Stage A]", "1 channel: noisy"],
  ["Patches x batch", "128 x 128 x 16", "64 x 64 x 64 (2 x 32, gradient accumulation)"],
  ["Pixels per step", HL("262,144"), HL("262,144")],
  ["Data mix", "50% Mayo / 50% LiTS train, ~30% tumour patches", "same (same seed, same patch list)"],
  ["Loss", "L1 + lesion-weighted L1 (tumour 5x, mask dilated 3 px)", "same"],
  ["Optimiser", "AdamW, lr 2e-4 → 1e-6 cosine, wd 1e-4, 30,000 iterations", "same"],
  ["Checkpoint", "Lowest 1,000-iteration training-loss window", "same (done: 0.004045)"],
  ["GPU", "A100 MIG 4g.20gb", "A100 MIG 4g.20gb (11 h 3 min)"],
], { x: 0.6, y: 1.4, w: 12.1, colW: [2.4, 5.0, 4.7], rowH: 0.42 });
note(s, "Everything except the network is identical, so a difference between the two isolates the refiner design.", 6.35);
s.addNotes("The control already finished on 19 Nov at 11:19. The refiner uses the same deterministic patch list (patch i depends only on seed and i).");

// ---- schedule
s = content("Training plan", "Day-by-day schedule to the deadline");
const days = [
  ["19 Nov", "Control trained (done)\nFree disk, start Stage A", C.accent6],
  ["20 Nov", "Stage A done\nTrain refiner\nEvaluate control", C.accent1],
  ["21 Nov", "Evaluate refiner\nFirst verdict on 10 patients", C.accent1],
  ["22 Nov", "Extended set conversion (CPU)\nScore CT-Mamba, refiner", C.accent2],
  ["23 Nov", "Score control, CTformer, CoreDiff on extended set", C.accent2],
  ["24 Nov", "Final statistics, figures, report tables", C.text2],
  ["25 Nov", "Buffer + final presentation", C.accent3],
];
const dw = 1.6, dg = 0.15;
days.forEach(([d, t, col], i) => {
  const x = 0.6 + i * (dw + dg);
  n++;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: dw, h: 0.7, rectRadius: 0.08, fill: { color: col }, line: { color: col }, objectName: `day${n}` });
  s.addText(d, { x, y: 1.5, w: dw, h: 0.7, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: `dayt${n}` });
  card(s, x, 2.35, dw, 2.4, null, t.split("\n"), { size: 14 });
});
card(s, 0.6, 5.0, 5.9, 1.25, "Milestone 1 (21 Nov)", "Steps 1-5 with a first evaluation on the existing 10 patients, within the 3-day target.", { size: 14 });
card(s, 6.8, 5.0, 5.9, 1.25, "Milestone 2 (24 Nov)", "Final verdict on the 20-patient, 106-tumour extended test set.", { size: 14 });
s.addNotes("If the GPU queue delays the refiner by more than a day, both refiner and control evaluations keep the same budget; the plan allows shortening both runs equally.");

// ---- safeguards
s = content("Training plan", "Safeguards built into training");
const guards = [
  ["No test leakage", "The patch sampler refuses Mayo L506 and every locked LiTS patient; checkpoints are chosen on training loss only."],
  ["Tests before GPU time", "26 checks (shapes, zero-init, loss inside/outside mask, sampling share, locked-patient refusal) + a smoke run in every job."],
  ["Masks only in the loss", "Inference takes only the CT slice; no tumour mask and no nnU-Net are ever used to produce an image."],
  ["Disk and queue", "Every job checks the 40 GB line first and stops cleanly; runs resume from latest.pth after any interruption."],
  ["Fixed in advance", "Success criteria (a)-(e) and the extended-set selection rule were committed before any result."],
  ["Stop rule", "A step that fails 3 times is logged as blocked and the plan moves on, so one problem cannot stop the week."],
];
guards.forEach(([h, b], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.45 + Math.floor(i / 3) * 2.45;
  card(s, x, y, 3.9, 2.25, h, b, { size: 14, headSize: 17 });
});
s.addNotes("These make the result trustworthy: if the refiner wins, it is not because it saw test data or because the metric was picked afterwards.");
}

// ================================================================= 18 conclusion
pres.addSection({ title: "Summary" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Summary" });
s.addText("Summary and next steps", { placeholder: "title" });
s.addText([
  { text: "Built a complete benchmark: 3 denoisers, real + simulated quarter dose, locked tests, a fixed tumour segmenter.", options: { bullet: true, breakLine: true } },
  { text: "CT-Mamba wins on images (29.42 dB Mayo, 29.92 dB LiTS) but has the lowest tumour Dice (0.543 vs 0.582 normal dose).", options: { bullet: true, breakLine: true } },
  { text: "Next: evaluate refiner vs control, score the 106-tumour extended set, verdict against the pre-registered criteria (a)-(e).", options: { bullet: true } },
], { x: 0.8, y: 3.9, w: 11.7, h: 2.6, fontSize: 18, color: C.background1, paraSpaceAfter: 8, valign: "top", isTextBox: true, objectName: "summary" });
s.addNotes("End on the question for the teacher: does the evaluation plan and the criteria look right to you?");

(async () => {
  const out = process.argv[2] || "LDCT_Progress_Review.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
