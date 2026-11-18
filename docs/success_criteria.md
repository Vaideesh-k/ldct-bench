# Success criteria for the tumour-aware refiner (fixed BEFORE any refiner/control result)

Committed on 18 Nov 2026, branch `final-week`, before any refiner or control model was trained
or scored. They are not changed after results come in; if a rule turns out to be ambiguous, the
reading below applies and any later note is marked as written after the results.

Models: **refiner** = `ct_mamba_refiner` (frozen CT-Mamba + 2D U-Net residual, tumour-weighted
loss); **control** = CT-Mamba fine-tuned with the same data mix, tumour-aware sampling, loss and
number of training pixels, without a refiner. **Baselines** = CT-Mamba (`best_from11.pth`),
CTformer, CoreDiff, as already trained. Normal dose and quarter dose are reference inputs, not
baselines.

The refiner WINS only if ALL five hold:

| | criterion | measured on | rule |
|---|---|---|---|
| a | image quality is kept | Mayo L506, 526 slices, window -160..240 HU | mean PSNR >= 29.42 dB (CT-Mamba) |
| b | tumour outlines at least as good as the best baseline | LiTS locked test set, per-tumour Dice (same nnU-Net, same scorer) | mean per-tumour Dice >= the highest of CT-Mamba / CTformer / CoreDiff |
| c | the refiner itself helps (not just the extra training) | same as b | mean per-tumour Dice of refiner > control |
| d | the differences are real | paired per-tumour Dice: refiner vs control and refiner vs best baseline | two-sided paired Wilcoxon p < 0.05 AND paired bootstrap 95% CI of the mean difference (10,000 resamples of tumours, seed 0) excludes 0 |
| e | no extra false alarms | LiTS locked test set | false positives per patient of the refiner <= that of CT-Mamba |

Readings fixed in advance:
- (a) and (e) are point estimates on the full test set; bootstrap CIs are reported for them too.
- (b) and (c) are point estimates; (d) is what makes (b) and (c) count as more than noise. If (b)
  or (c) holds but (d) does not, the result is reported as "not significant", i.e. a FAIL.
- Everything is evaluated first on the existing 10-patient set (34 tumours) and then on the
  extended set (step 6). The extended set is the final verdict; the 10-patient result is
  reported as preliminary.
- Detection counts (all / under 10 mm) are reported for every input but are not a criterion.
- Tumour masks are used only in the training loss. At test time the refiner and the control see
  only images; masks and the evaluation nnU-Net are never used at inference.
- The checkpoint is the one with the lowest training loss (fixed rule); no test or validation
  data choose it.
