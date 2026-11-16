# Morning report (updated 16 Nov 2026)

Branch **`autonomous-nov12`** (pushed; please review and merge). Full log: docs/PROGRESS.md.
The 12 Nov session was cut off around 14:30; on 16 Nov I resumed, collected what the
cluster had done meanwhile and fixed one problem in the LiTS simulation.

## Headline results (Mayo L506, 526 slices, window -160..240 HU, same scorer for all)

| model | PSNR dB | SSIM | RMSE HU | s/slice (A100) | training |
|---|---|---|---|---|---|
| quarter dose (input) | 24.47 | 0.8382 | 24.64 | | |
| **CT-Mamba** (job 27804) | **29.42** | 0.8776 | **13.85** | 1.058 | 250 epochs, 2 d 22 h |
| CTformer (job 35390) | 28.56 | 0.8680 | 15.28 | 0.112 | 530k updates, 12 h 22 min |
| CoreDiff (job 35391) | 29.28 | **0.8817** | 14.07 | 0.087 | 150k updates, 13 h 54 min |

Per-slice scores: dgx_results/<model>_scores_mayo_L506.csv. Training logs: dgx_results/*_train_log.csv.
## LiTS results (10 locked test patients, 1,458 liver slices, simulated quarter dose, same scorer)

| model | PSNR dB | SSIM | RMSE HU | s/slice (MIG 3g.20gb) |
|---|---|---|---|---|
| quarter dose (input) | 24.29 | 0.8193 | 25.77 | |
| **CT-Mamba** | **29.92** | **0.8608** | **13.53** | 2.320 |
| CTformer | 28.51 | 0.8354 | 15.71 | 0.253 |
| CoreDiff | 29.43 | 0.8560 | 14.08 | 0.181 |

Same ranking as on Mayo L506; none of the models saw a LiTS slice during training. Timing is on a
MIG slice, so it is not comparable with the A100 column above. Per-slice scores:
dgx_results/<model>_scores_lits.csv. Low dose = parallel-beam Poisson + electronic noise at
25 % exposure, exposure calibrated per patient to the Mayo quarter-dose liver noise (48.2 HU).

## What finished
- Steps 1, 2, 4, 6, 7 of your plan, as in the 12 Nov report (sync, both baselines wrapped +
  trained + scored, detection metrics, hybrid code/tests/measurements, lesion loss code/tests).
- Step 3: locked split, simulator, Mayo noise target (48.2 HU), all 51 part-1 patients
  converted (7,274 liver slices with masks, 5 patients without tumour).
- Step 5: nnU-Net v2 installed; dataset + training job written and queued (35522).

## Running / queued on the DGX (16 Nov)
| job | what | expected |
|---|---|---|
| 36228 | LiTS low-dose resimulation with per-patient exposure calibration (CPU) | DONE 16 Nov, 2 h 49 min, 51/51 patients |
| 36229 / 36230 / 36231 | predict + evaluate CTformer / CoreDiff / CT-Mamba on the 10 locked LiTS test patients (MIG slices) | DONE 16 Nov (8 min / 6 min / 58 min), table above |
| 35522 | nnU-Net 3d_fullres, 250 epochs, fold 0, then predict + detect_eval on the normal-dose test patients | pending in the full-A100 queue, SLURM's estimated start 20 Nov 10:36 (five A100 jobs ahead, one 5 days old); ~1 day once running |
| 36232 | after all of the above: nnU-Net on normal dose / quarter dose / CT-Mamba / CTformer / CoreDiff LiTS test outputs -> lesion detection rates (results/detect_summary.csv) | ~1-2 h on a MIG slice |

## Problems and what I did about them
- Kaggle serves this dataset only as a whole archive; downloaded on a CPU node, patients
  extracted one at a time. My first pipeline run deleted the archive after patient 0 and
  reported success; fixed and tested, re-downloaded, rebuilt (12 Nov evening).
- The single-exposure calibration gave 32-686 HU of added liver noise across patients
  (median 70) against the 37-62 HU of the real Mayo patients. The simulation now picks
  the exposure per patient (like a scanner's AEC) to hit 48.2 HU: job 36228 finished, all
  51 patients now sit at 45.9-51.9 HU (median 48.5; Mayo scorer, same uniform-tissue ROI)
  with exposures from 4.5e5 to 1.4e7 photons (configs/lits_sim.json, per-patient values and
  calibration history in dgx_results/lits_meta/<pid>/meta.json, summary in
  dgx_results/lits_added_noise.json). Documented limitation stays: parallel-beam geometry,
  no bow-tie filter.
- Hybrid variant "small" is 4x CT-Mamba's size (the 8x8 scale has the widest channels);
  docs/hybrid_design.md has the one-line fix if you want a lighter version.

## Decisions I made on my own (all logged in PROGRESS.md)
Official baseline schedules run in full with the shared cosine schedule and best.pth rule;
CoreDiff keeps edge slices; calibration on a train patient; per-patient exposure
calibration for LiTS; hybrid gate init 0.5; Idea-1 extra L1 scaled x100 for CT-Mamba;
nnU-Net 250-epoch trainer in a venv; LiTS test set = locked split only; LiTS archive deleted
after the build (re-downloadable in 50 min).

## What I need from you
1. Review/merge `autonomous-nov12`.
2. Idea 2: approve the variants (especially "small"), then submit the four 50-epoch runs
   (commands in the header of slurm/short50.sbatch, ~14 h each, chained on one A100).
3. Idea 1: submit slurm/finetune.sbatch (control, then lesion; commands in its header).
   The final low-dose LiTS slices it needs are in place (36228 done).
4. Nothing for the detection comparison: job 36232 does it automatically after nnU-Net
   and the LiTS predictions (results/detect_summary.csv on the DGX). If any job in the
   chain fails, the dependants stay pending with reason "DependencyNeverSatisfied";
   `scancel` them and resubmit after the fix (commands in each script's header).
