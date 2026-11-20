# Status report — final week, 20 Nov 2026 evening

Branch **`final-week`** (pushed after every step). Full log: docs/PROGRESS.md.

## Done
| step | status |
|---|---|
| 1 success criteria | **DONE** before any result (docs/success_criteria.md) |
| 2 Stage A | **DONE** inside job 42024, on node-local scratch (home stayed at 40 GB) |
| 3 refiner (2D) | **DONE**, 30,000 iterations |
| 3b refiner 2.5D (owner decision 20 Nov, before results) | **DONE**, 30,000 iterations |
| 4 control | **DONE** (job 39377) and evaluated |
| 5 first evaluation | **DONE**: dgx_results/final_existing10{,_25d}.md |
| 6 extended test set | split committed; conversion + evaluation **next** |
| 7 report material | deck updated with results; figures after step 6 |

## First evaluation (10 LiTS patients, 34 tumours; L506) — preliminary
| input | L506 PSNR | SSIM | detected | < 10 mm | FP/pt | per-tumour Dice |
|---|---|---|---|---|---|---|
| CT-Mamba | 29.42 | 0.8776 | 27/34 | 5/9 | 0.30 | 0.543 |
| CTformer | 28.56 | 0.8680 | 28/34 | 5/9 | 0.50 | 0.585 |
| CoreDiff | 29.28 | 0.8817 | 28/34 | 5/9 | 0.50 | 0.584 |
| control | 29.41 | 0.8775 | 28/34 | 5/9 | 0.70 | 0.582 |
| refiner 2D | 29.42 | 0.8775 | 28/34 | 5/9 | 0.50 | 0.582 |
| **refiner 2.5D** | **29.64** | **0.8826** | 28/34 | 5/9 | 0.60 | 0.585 |

Criteria a / b / c / d / e:
- refiner 2D: FAIL / FAIL / FAIL / FAIL / FAIL — **not a win** (it stays at CT-Mamba's output).
- refiner 2.5D: **PASS** / FAIL (by 0.0006) / PASS (not significant) / FAIL / FAIL — **not a win**.

Plain reading: the 2.5D refiner is the best denoiser of the project on image quality (+0.22 dB
over CT-Mamba, significant), but no denoiser changes tumour detection on this set: every trained
model finds the same 28 of 34 tumours and the same 5 of 9 small ones, and the Dice differences are
within noise.

## Next
1. Extended set (20 patients, 106 tumours, 58 < 10 mm): conversion on node scratch, then
   detection for all inputs incl. both refiners — the final verdict per the criteria.
2. Figures + report tables from final_stats.py on the extended set.
