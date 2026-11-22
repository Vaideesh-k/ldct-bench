# Status report — final week, 21 Nov 2026 night

Branch **`final-week`** (pushed after every step). Full log: docs/PROGRESS.md.

## Steps
| step | status |
|---|---|
| 1 success criteria | DONE before any result |
| 2 Stage A | DONE (node-local scratch) |
| 3 refiner 2D + 2.5D variant | DONE |
| 4 control | DONE |
| 5 first evaluation (10 patients) | DONE: dgx_results/final_existing10{,_25d}.md |
| 6 extended set (20 patients, 106 tumours, 57 < 10 mm) | **DONE**: dgx_results/final_extended20{,_25d}.md |
| 7 report material | **DONE**: 4 figures in docs/report/figures, deck 37 slides |

## Final result (20 LiTS patients, 106 tumours; Mayo L506)
| input | L506 PSNR | detected | < 10 mm | FP/pt | per-tumour Dice |
|---|---|---|---|---|---|
| normal dose | - | 70/106 | 26/57 | 0.40 | 0.417 |
| quarter dose | 24.47 | 60/106 | 18/57 | 0.35 | 0.365 |
| CT-Mamba | 29.42 | 64/106 | 20/57 | 0.30 | 0.354 |
| CTformer | 28.56 | 68/106 | 23/57 | 0.40 | 0.416 |
| CoreDiff | 29.28 | 64/106 | 19/57 | 0.40 | 0.378 |
| **control** | 29.41 | **71/106** | **26/57** | 0.65 | **0.417** |
| refiner 2D | 29.42 | 69/106 | 24/57 | 0.50 | 0.399 |
| refiner 2.5D | **29.64** | 68/106 | 23/57 | 0.50 | 0.371 |

**Verdict: NOT A WIN** for both refiners (2D fails a-e; 2.5D passes only a).
- The fine-tuned control (CT-Mamba + LiTS data + tumour-weighted loss) recovers normal-dose
  tumour Dice and small-tumour detection; the refiner design adds nothing (2D is significantly
  below the control, p 0.0007).
- The 2.5D refiner has the best PSNR but a significantly lower Dice than the control, CTformer
  and normal dose: better image scores again do not mean better tumour outlines.

## Issues found and handled
- LiTS masks of volumes 48-130 have an identity NIfTI header; conversion flipped them left-right.
  Fixed before any extended-set use. Training patients lits_048-050 (3 of 41) had flipped masks
  in nnU-Net and refiner/control training; not retrained, reported as a limitation.
  All test patients verified aligned.

## Next
- Nothing queued. All brief steps done; write-up of the report text from these tables and figures.
