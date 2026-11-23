# Success criteria for the edge-preserving fine-tune (fixed BEFORE it is trained)

Written 23 Nov 2026, branch `final-week`, before `ct_mamba_edge` was trained or scored. Not changed
after results come in.

## Why this model (from the 20-patient findings)
1. Fine-tuning CT-Mamba on LiTS with the tumour-weighted loss (the control) is what raised
   tumour Dice (+0.063 vs CT-Mamba, p < 1e-4). Adding a network on top (refiners) did not.
2. The models that smooth most (CT-Mamba, CoreDiff, 2.5D refiner) have the lowest Dice even with
   the highest PSNR; CTformer keeps texture and has near-normal-dose Dice at the lowest PSNR.
3. The 2.5D refiner reached the lowest training loss and a worse test Dice: pixel L1 is satisfied
   by a smooth image and does not measure whether tumour edges survive.

So the architecture stays CT-Mamba, and only the objective changes: **edge model =
the control's exact setup** (same init `best_from11.pth`, data, sampling, 30,000 iterations,
lr, batch, checkpoint rule) **+ 1.0 x lesion-weighted L1 on the horizontal and vertical image
gradients** (`gradient_lesion` in ldct/tumour_train.py). The weight 1.0 is fixed here and not
tuned. Since the only difference from the control is this term, edge minus control measures it.

## Test data
- **Primary: `lits_fresh`**, 10 LiTS part-2 patients never used for anything
  (configs/lits_fresh_split.json, rule fixed in that file; scan count 240 tumours, 137 under
  10 mm). The 20-patient set was seen while designing this model, so it is secondary only.
- Mayo L506 for image quality, as before.
- Same nnU-Net, same scorer, same window (-160..240 HU), same bootstrap (10,000, seed 0).

## The edge model WINS only if ALL hold
| | criterion | rule |
|---|---|---|
| a | better image quality than every baseline on Mayo | L506 mean PSNR >= 29.42 dB (CT-Mamba, the best baseline) |
| b | better image quality than every baseline on LiTS | lits_fresh mean PSNR > the best of CT-Mamba / CTformer / CoreDiff |
| c | better tumour Dice than every baseline | lits_fresh per-tumour Dice > the best baseline's, paired Wilcoxon p < 0.05 AND paired bootstrap 95% CI excludes 0 |
| d | the edge term itself helps | lits_fresh per-tumour Dice > control, same significance test |

Reported, not criteria: false positives per patient (the control had more than CT-Mamba),
detection counts, Dice under 10 mm, per-patient Dice, and the 20-patient results.
Masks are used only in the training loss; the edge model sees only images at test time.
Checkpoint = lowest training window loss (fixed rule); no test data choose it.
