# Edge model on the fresh test set (criteria: docs/success_criteria_edge.md)

| input | L506 PSNR | fresh PSNR | detected | < 10 mm | FP/pt | per-tumour Dice [95% CI] | Dice < 10 mm |
|---|---|---|---|---|---|---|---|
| normal dose | - | - | 132/240 | 39/137 | 0.80 | 0.349 [0.302, 0.394] | 0.167 |
| quarter dose | 24.47 | 23.98 | 124/240 | 34/137 | 1.00 | 0.330 [0.284, 0.375] | 0.140 |
| CT-Mamba | 29.42 | 29.96 | 116/240 | 36/137 | 0.80 | 0.260 [0.220, 0.301] | 0.138 |
| CTformer | 28.56 | 28.77 | 125/240 | 38/137 | 1.00 | 0.323 [0.278, 0.368] | 0.161 |
| CoreDiff | 29.28 | 29.60 | 122/240 | 40/137 | 0.90 | 0.297 [0.253, 0.339] | 0.158 |
| control | 29.41 | 31.22 | 132/240 | 43/137 | 1.10 | 0.334 [0.289, 0.378] | 0.174 |
| refiner | 29.42 | 31.00 | 124/240 | 36/137 | 1.00 | 0.322 [0.277, 0.367] | 0.158 |
| 2.5D refiner | 29.64 | 31.62 | 122/240 | 36/137 | 1.00 | 0.321 [0.278, 0.366] | 0.156 |
| edge model | 29.41 | 31.23 | 132/240 | 43/137 | 1.10 | 0.333 [0.288, 0.379] | 0.174 |

## Criteria

- (a) L506 PSNR >= 29.42 dB (CT-Mamba): 29.4147 -> FAIL
- (b) lits_fresh PSNR > best baseline (CT-Mamba 29.96): 31.2282 -> PASS
- (c) Dice > best baseline (CTformer 0.3227), significant: 0.3334 -> PASS
- (d) Dice > control (0.3335), significant: 0.3334 -> FAIL

**Verdict: NOT A WIN**

## Paired per-tumour Dice, edge minus other

- vs control: -0.0002 [-0.0010, +0.0007], Wilcoxon p 0.41
- vs CT-Mamba: +0.0734 [+0.0559, +0.0924], Wilcoxon p 5.2e-18 (significant)
- vs CTformer: +0.0106 [+0.0002, +0.0219], Wilcoxon p 0.028 (significant)
- vs CoreDiff: +0.0367 [+0.0213, +0.0526], Wilcoxon p 9.1e-09 (significant)
- vs refiner: +0.0117 [+0.0044, +0.0198], Wilcoxon p 0.0091 (significant)
- vs 2.5D refiner: +0.0121 [+0.0016, +0.0216], Wilcoxon p 0.0012 (significant)
- vs normal dose: -0.0152 [-0.0310, -0.0001], Wilcoxon p 0.0041 (significant)
