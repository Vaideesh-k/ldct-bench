# Results: existing10 (candidate: refiner)

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

## LiTS detection (10 patients), same nnU-Net

| input | detected | < 10 mm detected | FP / patient | per-tumour Dice | Dice < 10 mm |
|---|---|---|---|---|---|
| normal dose | 27/34 | 5/9 | 0.60 [0.20, 1.00] | 0.582 [0.459, 0.697] | 0.299 |
| quarter dose | 28/34 | 5/9 | 0.50 [0.20, 0.80] | 0.565 [0.443, 0.677] | 0.276 |
| CT-Mamba | 27/34 | 5/9 | 0.30 [0.00, 0.60] | 0.543 [0.425, 0.654] | 0.248 |
| CTformer | 28/34 | 5/9 | 0.50 [0.10, 0.90] | 0.585 [0.461, 0.700] | 0.307 |
| CoreDiff | 28/34 | 5/9 | 0.50 [0.00, 1.10] | 0.584 [0.466, 0.699] | 0.310 |
| control | 28/34 | 5/9 | 0.70 [0.20, 1.30] | 0.582 [0.463, 0.695] | 0.311 |
| refiner | 28/34 | 5/9 | 0.50 [0.10, 0.90] | 0.582 [0.462, 0.693] | 0.306 |
| 2.5D refiner | 28/34 | 5/9 | 0.60 [0.20, 1.00] | 0.585 [0.467, 0.696] | 0.306 |

## Paired comparisons (difference = first - second)

| comparison | mean diff [95% CI] | Wilcoxon p | significant |
|---|---|---|---|
| dice_ct_mamba_refiner_vs_ct_mamba_control | -0.0002 [-0.0041, +0.0038] | 0.5847 | no |
| fp_ct_mamba_refiner_vs_ct_mamba_control | -0.2000 [-0.5000, +0.0000] | 0.5000 | no |
| dice_ct_mamba_refiner_vs_ct_mamba_refiner25d | -0.0027 [-0.0063, +0.0005] | 0.2645 | no |
| fp_ct_mamba_refiner_vs_ct_mamba_refiner25d | -0.1000 [-0.4000, +0.2000] | 1.0000 | no |
| dice_ct_mamba_refiner_vs_ct_mamba | +0.0394 [+0.0092, +0.0797] | 0.0168 | yes |
| fp_ct_mamba_refiner_vs_ct_mamba | +0.2000 [+0.0000, +0.5000] | 0.5000 | no |
| dice_ct_mamba_refiner_vs_ctformer | -0.0033 [-0.0101, +0.0033] | 0.4662 | no |
| fp_ct_mamba_refiner_vs_ctformer | +0.0000 [+0.0000, +0.0000] | 1.0000 | no |
| dice_ct_mamba_refiner_vs_corediff | -0.0025 [-0.0072, +0.0021] | 0.2949 | no |
| fp_ct_mamba_refiner_vs_corediff | +0.0000 [-0.3000, +0.3000] | 1.0000 | no |
| dice_ct_mamba_refiner_vs_quarter | +0.0170 [+0.0000, +0.0377] | 0.2563 | no |
| fp_ct_mamba_refiner_vs_quarter | +0.0000 [-0.4000, +0.4000] | 1.0000 | no |
| dice_ct_mamba_refiner_vs_full | +0.0001 [-0.0126, +0.0143] | 0.7156 | no |
| fp_ct_mamba_refiner_vs_full | -0.1000 [-0.4000, +0.2000] | 1.0000 | no |
| psnr_ct_mamba_refiner_vs_quarter_dose | +4.9488 [+4.9248, +4.9726] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_quarter_dose | +0.0393 [+0.0389, +0.0398] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_quarter_dose | -10.7862 [-11.0754, -10.5091] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ct_mamba | -0.0011 [-0.0012, -0.0010] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ct_mamba | -0.0001 [-0.0001, -0.0001] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_ct_mamba | +0.0017 [+0.0015, +0.0018] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ctformer | +0.8561 [+0.8504, +0.8620] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ctformer | +0.0095 [+0.0094, +0.0096] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_ctformer | -1.4284 [-1.4569, -1.4006] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_corediff | +0.1387 [+0.1313, +0.1462] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_corediff | -0.0042 [-0.0045, -0.0040] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_corediff | -0.2194 [-0.2300, -0.2089] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ct_mamba_control | +0.0032 [+0.0026, +0.0038] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ct_mamba_control | -0.0000 [-0.0000, +0.0000] | 0.0027 | no |
| rmse_ct_mamba_refiner_vs_ct_mamba_control | -0.0062 [-0.0074, -0.0051] | 0.0000 | yes |
| psnr_ct_mamba_refiner_vs_ct_mamba_refiner25d | -0.2196 [-0.2231, -0.2161] | 0.0000 | yes |
| ssim_ct_mamba_refiner_vs_ct_mamba_refiner25d | -0.0050 [-0.0051, -0.0050] | 0.0000 | yes |
| rmse_ct_mamba_refiner_vs_ct_mamba_refiner25d | +0.3364 [+0.3316, +0.3413] | 0.0000 | yes |
| dice_ct_mamba_control_vs_ct_mamba | +0.0395 [+0.0089, +0.0808] | 0.0588 | no |

## Success criteria (docs/success_criteria.md)

- (a) L506 PSNR >= 29.42 (refiner 29.4176): **FAIL**
- (b) per-tumour Dice >= best baseline (CTformer 0.5852) (refiner 0.5819): **FAIL**
- (c) per-tumour Dice > control (0.5820) (refiner 0.5819): **FAIL**
- (d) refiner vs control AND vs best baseline: Wilcoxon p<0.05 and CI excludes 0, in the refiner's favour: **FAIL**
- (e) FP/patient <= CT-Mamba (0.30) (refiner 0.5000): **FAIL**

**Verdict: NOT A WIN**
