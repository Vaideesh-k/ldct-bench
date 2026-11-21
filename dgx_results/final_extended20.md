# Results: extended20 (candidate: refiner)

## Mayo L506 (window -160..240 HU), mean [95% bootstrap CI]

| input | PSNR dB | SSIM | RMSE HU |
|---|---|---|---|
| quarter dose | 24.47 [24.29, 24.64] | 0.8382 [0.8345, 0.8417] | 24.64 [24.07, 25.20] |
| CT-Mamba | 29.42 [29.26, 29.57] | 0.8776 [0.8743, 0.8808] | 13.85 [13.58, 14.13] |
| CTformer | 28.56 [28.40, 28.71] | 0.8680 [0.8648, 0.8713] | 15.28 [14.98, 15.59] |
| CoreDiff | 29.28 [29.12, 29.43] | 0.8817 [0.8786, 0.8847] | 14.07 [13.80, 14.35] |
| control | 29.41 [29.26, 29.57] | 0.8775 [0.8743, 0.8806] | 13.86 [13.58, 14.13] |
| refiner | 29.42 [29.26, 29.57] | 0.8775 [0.8742, 0.8807] | 13.85 [13.57, 14.13] |
| 2.5D refiner | 29.64 [29.48, 29.79] | 0.8826 [0.8794, 0.8857] | 13.51 [13.24, 13.78] |

## LiTS detection (20 patients), same nnU-Net

| input | detected | < 10 mm detected | FP / patient | per-tumour Dice | Dice < 10 mm |
|---|---|---|---|---|---|
| normal dose | 70/106 | 26/57 | 0.40 [0.15, 0.65] | 0.417 [0.348, 0.486] | 0.220 |
| quarter dose | 60/106 | 18/57 | 0.35 [0.15, 0.55] | 0.365 [0.296, 0.438] | 0.170 |
| CT-Mamba | 64/106 | 20/57 | 0.30 [0.10, 0.50] | 0.354 [0.288, 0.421] | 0.175 |
| CTformer | 68/106 | 23/57 | 0.40 [0.15, 0.70] | 0.416 [0.346, 0.485] | 0.222 |
| CoreDiff | 64/106 | 19/57 | 0.40 [0.10, 0.80] | 0.378 [0.311, 0.448] | 0.193 |
| control | 71/106 | 26/57 | 0.65 [0.35, 1.00] | 0.417 [0.350, 0.484] | 0.231 |
| refiner | 69/106 | 24/57 | 0.50 [0.20, 0.80] | 0.399 [0.330, 0.469] | 0.201 |
| 2.5D refiner | 68/106 | 23/57 | 0.50 [0.25, 0.80] | 0.371 [0.302, 0.441] | 0.173 |

## Paired comparisons (difference = first - second)

| comparison | mean diff [95% CI] | Wilcoxon p | significant |
|---|---|---|---|
| dice_ct_mamba_refiner_vs_ct_mamba_control | -0.0177 [-0.0289, -0.0075] | 0.0007 | yes |
| fp_ct_mamba_refiner_vs_ct_mamba_control | -0.1500 [-0.3000, +0.0000] | 0.0833 | no |
| dice_ct_mamba_refiner_vs_ct_mamba_refiner25d | +0.0277 [+0.0147, +0.0426] | 0.0007 | yes |
| fp_ct_mamba_refiner_vs_ct_mamba_refiner25d | +0.0000 [-0.2500, +0.2500] | 1.0000 | no |
| dice_ct_mamba_refiner_vs_ct_mamba | +0.0451 [+0.0246, +0.0680] | 0.0000 | yes |
| fp_ct_mamba_refiner_vs_ct_mamba | +0.2000 [+0.0500, +0.4000] | 0.0455 | yes |
| dice_ct_mamba_refiner_vs_ctformer | -0.0167 [-0.0366, +0.0011] | 0.1745 | no |
| fp_ct_mamba_refiner_vs_ctformer | +0.1000 [+0.0000, +0.2500] | 0.1573 | no |
| dice_ct_mamba_refiner_vs_corediff | +0.0207 [-0.0000, +0.0417] | 0.0320 | no |
| fp_ct_mamba_refiner_vs_corediff | +0.1000 [-0.1000, +0.3000] | 0.3173 | no |
| dice_ct_mamba_refiner_vs_quarter | +0.0343 [+0.0072, +0.0626] | 0.0160 | yes |
| fp_ct_mamba_refiner_vs_quarter | +0.1500 [-0.1500, +0.5000] | 0.3657 | no |
| dice_ct_mamba_refiner_vs_full | -0.0179 [-0.0411, +0.0051] | 0.1297 | no |
| fp_ct_mamba_refiner_vs_full | +0.1000 [-0.1500, +0.3500] | 0.4142 | no |
| psnr_ct_mamba_refiner_vs_quarter_dose | +4.9488 [+4.9252, +4.9720] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_quarter_dose | +0.0393 [+0.0389, +0.0398] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_quarter_dose | -10.7862 [-11.0741, -10.5081] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ct_mamba | -0.0011 [-0.0012, -0.0010] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ct_mamba | -0.0001 [-0.0001, -0.0001] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_ct_mamba | +0.0017 [+0.0015, +0.0018] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ctformer | +0.8561 [+0.8504, +0.8620] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ctformer | +0.0095 [+0.0094, +0.0096] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_ctformer | -1.4284 [-1.4573, -1.4010] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_corediff | +0.1387 [+0.1315, +0.1462] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_corediff | -0.0042 [-0.0045, -0.0040] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_corediff | -0.2194 [-0.2300, -0.2089] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ct_mamba_control | +0.0032 [+0.0026, +0.0038] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ct_mamba_control | -0.0000 [-0.0000, +0.0000] | 0.0027 | no |
| rmse_ct_mamba_refiner_vs_ct_mamba_control | -0.0062 [-0.0074, -0.0051] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ct_mamba_refiner25d | -0.2196 [-0.2231, -0.2161] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ct_mamba_refiner25d | -0.0050 [-0.0051, -0.0050] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_ct_mamba_refiner25d | +0.3364 [+0.3314, +0.3414] | 0.0000 | yes |
| dice_ct_mamba_control_vs_ct_mamba | +0.0628 [+0.0391, +0.0893] | 0.0000 | yes |

## Success criteria (docs/success_criteria.md)

- (a) L506 PSNR >= 29.42 (refiner 29.4176): **FAIL**
- (b) per-tumour Dice >= best baseline (CTformer 0.4159) (refiner 0.3992): **FAIL**
- (c) per-tumour Dice > control (0.4169) (refiner 0.3992): **FAIL**
- (d) refiner vs control AND vs best baseline: Wilcoxon p<0.05 and CI excludes 0, in the refiner's favour: **FAIL**
- (e) FP/patient <= CT-Mamba (0.30) (refiner 0.5000): **FAIL**

**Verdict: NOT A WIN**
