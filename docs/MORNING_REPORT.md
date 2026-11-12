# Morning report (session of 12 Nov 2026, cut short by the usage limit)

The session stopped early at ~09:15 because the Claude usage limit was reached.
Steps 1 and 2 are finished; step 3 is half done; steps 4-7 were not started.
Everything is on branch `autonomous-nov12` (pushed); details in docs/PROGRESS.md.

## Finished
- **Sync**: slurm/ scripts recovered from the DGX and committed; DGX code is current;
  ~/wheels deleted, conda cache cleaned (16 GB used of 50).
- **CTformer and CoreDiff** wrapped with their official settings (differences listed in
  the wrapper docstrings), pipeline extended (slice context, patch augmentation,
  per-slice scores), 28 new tests + old tests pass on Mac and DGX. Smoke tests on a
  MIG slice passed: CTformer ~1 day, CoreDiff ~1.4 days.
- **LiTS**: locked split committed (`configs/lits_split.json`, seed 0: 41 train /
  10 test of part 1, test = [2, 4, 10, 11, 22, 26, 28, 29, 32, 45]); low-dose simulator
  `scripts/make_lits_lowdose.py` (projection-domain Poisson + electronic noise,
  parallel-beam, documented as a limitation) with tests; `scripts/calibrate_lits_noise.py`
  and the download/convert loop `scripts/prep_lits.py` + `slurm/lits_prep.sbatch`.

## Running on the DGX
| job | what | state at 09:15 |
|---|---|---|
| 35390 | CTformer full run (train + predict + evaluate L506) | pending, 24 jobs ahead in gpu_student |
| 35391 | CoreDiff full run, `--dependency=afterany:35390` | pending |
| srun `mayo_noise` | Mayo quarter-minus-full noise in uniform soft tissue -> results/mayo_added_noise.json | was running (interactive srun; may have died with the session: rerun if the json is missing) |
| srun `lits_trial` | patient 0 converted with trial I0 = 2e5 into data/lits_trial | same |

## Not done (in order)
1. Finish LiTS calibration: read the Mayo median added-noise std (HU) and the trial
   patient's liver added-noise std from data/lits_trial/lits_000/meta.json, set
   `I0 = 2e5 * (std_trial / std_mayo)^2`, record it in docs/PROGRESS.md, delete
   data/lits_trial and data/lits_raw_trial, then
   `sbatch slurm/lits_prep.sbatch <I0> 0 2 4 10 11 22 26 28 29 32 45 <remaining train ids>`.
2. Step 4 `scripts/detect_eval.py` + tests.
3. Step 5 nnU-Net venv, dataset conversion, chained training job.
4. Step 6 hybrid design doc + `ct_mamba_hybrid` model + tests + unsubmitted sbatch scripts.
5. Step 7 lesion-weighted loss + tumour-aware sampling + tests + unsubmitted scripts.

## Decisions I made
- Official schedules kept in full (both under the 5-day cap).
- Calibration patient = train patient 0 (not a locked test patient).
- Both long jobs on the full A100 as per your default; CTformer (3.3 GiB) would start
  days earlier on a MIG slice if you want to change that: `scancel 35390 35391`, then
  `sbatch --job-name=ctformer --gres=gpu:a100_3g.20gb:1 slurm/train_model.sbatch ctformer`.

## Needed from you
- Review/merge `autonomous-nov12`.
- Decide whether to move CTformer to a MIG slice (above).
- Restart the session to continue with item 1 of "Not done".
