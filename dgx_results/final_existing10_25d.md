# Results: existing10 (candidate: 2.5D refiner)

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
| dice_ct_mamba_refiner25d_vs_ct_mamba_control | +0.0026 [-0.0030, +0.0085] | 0.5237 | no |
| fp_ct_mamba_refiner25d_vs_ct_mamba_control | -0.1000 [-0.4000, +0.2000] | 1.0000 | no |
| dice_ct_mamba_refiner25d_vs_ct_mamba_refiner | +0.0027 [-0.0005, +0.0063] | 0.2645 | no |
| fp_ct_mamba_refiner25d_vs_ct_mamba_refiner | +0.1000 [-0.2000, +0.4000] | 1.0000 | no |
| dice_ct_mamba_refiner25d_vs_ct_mamba | +0.0421 [+0.0104, +0.0843] | 0.0228 | yes |
| fp_ct_mamba_refiner25d_vs_ct_mamba | +0.3000 [+0.0000, +0.6000] | 0.2500 | no |
| dice_ct_mamba_refiner25d_vs_ctformer | -0.0006 [-0.0071, +0.0060] | 0.7327 | no |
| fp_ct_mamba_refiner25d_vs_ctformer | +0.1000 [-0.2000, +0.4000] | 1.0000 | no |
| dice_ct_mamba_refiner25d_vs_corediff | +0.0003 [-0.0057, +0.0065] | 0.7499 | no |
| fp_ct_mamba_refiner25d_vs_corediff | +0.1000 [-0.2000, +0.4000] | 1.0000 | no |
| dice_ct_mamba_refiner25d_vs_quarter | +0.0197 [+0.0038, +0.0396] | 0.1221 | no |
| fp_ct_mamba_refiner25d_vs_quarter | +0.1000 [-0.2000, +0.4000] | 1.0000 | no |
| dice_ct_mamba_refiner25d_vs_full | +0.0028 [-0.0077, +0.0156] | 0.8199 | no |
| fp_ct_mamba_refiner25d_vs_full | +0.0000 [-0.3000, +0.3000] | 1.0000 | no |
| psnr_ct_mamba_refiner25d_vs_quarter_dose | +5.1685 [+5.1471, +5.1892] | 0.0000 | yes |
| ssim_ct_mamba_refiner25d_vs_quarter_dose | +0.0444 [+0.0439, +0.0449] | 0.0000 | yes |
| rmse_ct_mamba_refiner25d_vs_quarter_dose | -11.1227 [-11.4146, -10.8437] | 0.0000 | yes |
| psnr_ct_mamba_refiner25d_vs_ct_mamba | +0.2186 [+0.2151, +0.2220] | 0.0000 | yes |
| ssim_ct_mamba_refiner25d_vs_ct_mamba | +0.0050 [+0.0049, +0.0050] | 0.0000 | yes |
| rmse_ct_mamba_refiner25d_vs_ct_mamba | -0.3348 [-0.3396, -0.3298] | 0.0000 | yes |
| psnr_ct_mamba_refiner25d_vs_ctformer | +1.0758 [+1.0691, +1.0824] | 0.0000 | yes |
| ssim_ct_mamba_refiner25d_vs_ctformer | +0.0145 [+0.0144, +0.0147] | 0.0000 | yes |
| rmse_ct_mamba_refiner25d_vs_ctformer | -1.7648 [-1.7953, -1.7354] | 0.0000 | yes |
| psnr_ct_mamba_refiner25d_vs_corediff | +0.3584 [+0.3518, +0.3652] | 0.0000 | yes |
| ssim_ct_mamba_refiner25d_vs_corediff | +0.0008 [+0.0006, +0.0011] | 0.0000 | yes |
| rmse_ct_mamba_refiner25d_vs_corediff | -0.5558 [-0.5656, -0.5462] | 0.0000 | yes |
| psnr_ct_mamba_refiner25d_vs_ct_mamba_control | +0.2229 [+0.2196, +0.2261] | 0.0000 | yes |
| ssim_ct_mamba_refiner25d_vs_ct_mamba_control | +0.0050 [+0.0050, +0.0051] | 0.0000 | yes |
| rmse_ct_mamba_refiner25d_vs_ct_mamba_control | -0.3427 [-0.3478, -0.3375] | 0.0000 | yes |
| psnr_ct_mamba_refiner25d_vs_ct_mamba_refiner | +0.2196 [+0.2161, +0.2231] | 0.0000 | yes |
| ssim_ct_mamba_refiner25d_vs_ct_mamba_refiner | +0.0050 [+0.0050, +0.0051] | 0.0000 | yes |
| rmse_ct_mamba_refiner25d_vs_ct_mamba_refiner | -0.3364 [-0.3413, -0.3316] | 0.0000 | yes |
| dice_ct_mamba_control_vs_ct_mamba | +0.0395 [+0.0089, +0.0808] | 0.0588 | no |

## Success criteria (docs/success_criteria.md)

- (a) L506 PSNR >= 29.42 (refiner 29.6372): **PASS**
- (b) per-tumour Dice >= best baseline (CTformer 0.5852) (refiner 0.5846): **FAIL**
- (c) per-tumour Dice > control (0.5820) (refiner 0.5846): **PASS**
- (d) refiner vs control AND vs best baseline: Wilcoxon p<0.05 and CI excludes 0, in the refiner's favour: **FAIL**
- (e) FP/patient <= CT-Mamba (0.30) (refiner 0.6000): **FAIL**

**Verdict: NOT A WIN**
