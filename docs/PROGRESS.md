# Progress log (autonomous session, 12 Nov 2026)

Branch: `autonomous-nov12` (pushed after every step; owner reviews and merges).
Decisions taken by the owner before the session: no teammates on CTformer/CoreDiff;
Idea 2 (hybrid) approved for code + tests only; Idea 3 not started; no full runs of
new models; defaults for queue/schedule/LiTS/storage/nnU-Net questions (see below).

## Step 1: SYNC (done, ~09:00)

- DGX copy of the repo was an rsync (no .git). Code-only files that existed only on the
  DGX: `slurm/` (8 files) -> copied back, committed with `slurm/README.md` (1c9091a).
  `tests/check_pipeline.py`, `ldct/models/ct_mamba.py`, `scripts/train.py` were simply
  older on the DGX (no DGX-side edits lost).
- Latest code rsynced to the DGX (excluding data/, runs/, results/, logs/, .git/,
  third_party/); md5 of train.py, ct_mamba.py, check_train.py match.
- DGX state: ctmamba env OK (torch 2.7.1), kaggle CLI 1.7.4.5 with credentials, no jobs
  of ours running; sacct shows only the finished 27804 (train) and 27866 (watcher).
- data/mayo slice counts (quarter = full): L067 560, L096 823, L109 318, L143 585,
  L192 600, L286 525, L291 856, L310 533, L333 610, L506 526 (5,410 training slices).
- Storage: 17 GB before; deleted `~/wheels` (834 MB) and ran `conda clean` (approved
  default) -> 16 GB used. Budget: never below 10 GB free, i.e. at most ~40 GB used.
- Queue: gpu_student has ~17 pending full-A100 jobs from other users ahead of anything
  we submit (some running 13 h+ with 7-day limits). Expect days of waiting for full
  A100 jobs. MIG 3g.20gb slices looked free.
- Compute nodes have internet (kaggle.com and pypi.org reachable from cpu_student), so
  Kaggle downloads can run inside the conversion job.
- LiTS on Kaggle: part 1 = volumes 0-50 (51 patients, 9.7 GB raw) + all 131 masks;
  part 2 = volumes 51-130.

## Step 2: BASELINES CTformer + CoreDiff (done, jobs submitted ~09:00-09:10)

- Official repos pinned: CTformer 3c62f58 (wdayang/CTformer, last commit Apr 2022),
  CoreDiff 9bc31dc (qgao21/CoreDiff, Jan 2025) via scripts/get_ctformer.sh / get_corediff.sh.
- Wrappers: `ldct/models/ctformer.py`, `ldct/models/corediff.py`; configs in configs/models/.
  Settings copied from each paper's official code (documented line by line in the
  wrapper docstrings). Fixed rules win where they conflict; every difference is listed
  in the docstrings. Summary of differences:
  - CTformer: our 1 mm data (official README: 3 mm); shared cosine lr schedule instead
    of "halve every 8000 iterations"; best.pth by lowest train loss instead of the last
    iteration; tiles averaged instead of centre-cropped at test time. Update count
    matched to the released checkpoint: 530,000 steps = 1565 epochs x 339 steps.
  - CoreDiff: best.pth = EMA weights of the lowest-train-loss epoch instead of the EMA
    at iteration 150k; edge slices kept (neighbour repeated) so all 526 L506 slices are
    scored; EMA update lags one step. 150,294 steps = 222 epochs x 677 steps.
- Pipeline additions (defaults unchanged, CT-Mamba unaffected): `data.context` (adjacent
  slices as input channels, needed by CoreDiff), `data.augment` (CTformer's flip/rotate
  of half the patches), evaluate.py now writes per-slice `scores_<set>.csv`.
- Tests: tests/check_baselines.py (28 checks) + the 7 old test files all pass on the Mac
  (CPU) and on the DGX.
- Smoke + timing on a MIG 3g.20gb slice (jobs 35388, 35389, both SMOKE_OK):
  | model | params | batch | s/step | peak mem | est. full run |
  |---|---|---|---|---|---|
  | ctformer | 1.45 M trainable (1.65 M incl. fixed tables) | 16 slices x 4 = 64 patches | 0.166 | 3.3 GiB | ~25 h (MIG), less on a full A100 |
  | corediff | 4.83 M (+ EMA copy) | 8 whole slices (timed at 2, scaled) | ~0.82 (est.) | ~14.5 GiB (est.) | ~34 h (MIG), less on a full A100 |
  Both far under the 5-day cap, so the official schedules are run in full (Q2 not needed).
- SUBMITTED (Q1 default: full A100, chained): **ctformer = job 35390**, **corediff = job
  35391** (`--dependency=afterany:35390`). Each runs scripts/run_all.sh (train, predict
  on L506 with best.pth, evaluate with per-slice scores, baseline row). At submission
  time 24 jobs were pending in gpu_student ahead of ours; expect a wait of days.

## Step 3: LiTS data (in progress)

- Locked split committed: `configs/lits_split.json` (seed 0, 20 % test): test patients
  [2, 4, 10, 11, 22, 26, 28, 29, 32, 45], 41 train, all from part 1 (volumes 0-50).
- `scripts/make_lits_lowdose.py`: NIfTI -> our .npy layout (full / quarter / mask, int16 HU,
  radiological orientation, liver range +/- 5 slices) with projection-domain noise
  insertion (parallel-beam skimage radon/iradon, Poisson + electronic noise, Yu et al.
  2012 style; parallel-beam/no bow-tie documented as a limitation). Tests: tests/check_lits.py.
- **Mayo calibration target** (`scripts/calibrate_lits_noise.py --mayo`, 276 slices of the
  9 training patients, uniform liver-like soft tissue): std(quarter - full) median
  **48.2 HU** (IQR 41.3-54.6; per patient 37-62), full-dose std there 23.7 HU.
  Saved on the DGX in results/mayo_added_noise.json.
- PROBLEM + workaround: Kaggle per-file download (`kaggle datasets download -f ...`) returns
  404 for every file of this dataset (CLI 1.7.4.5, the REST endpoint with and without a
  version number, login and compute node; 3 attempts each). Only the whole part-1 archive
  can be fetched, and the CLI/SDK buffer it in RAM, so: **job 35430** downloads the 5.2 GB
  archive on cpu_student (16 GB RAM), then **job 35431** (afterok) runs
  scripts/lits_pipeline.sh: trial conversion of TRAIN patient 0 at I0 = 2e5, calibration
  I0 = 2e5 x (std_trial / 48.23)^2 written to configs/lits_sim.json, then all 51 patients
  (0, the 10 locked test patients, then the train patients) extracted ONE at a time from
  the archive (never unpacked as a whole), raw files deleted after each, disk logged every
  10 patients, stop above 38 GB used. Afterwards results/lits_added_noise.json reports the
  achieved liver noise of lits_000/002/004 against the target. The archive (5.2 GB) stays
  until the pipeline is done; delete it afterwards to free space.
- Download job 35430 COMPLETED in 24 min: 5.19 GB, 182 members (51 volumes + 131 masks),
  `testzip` clean. Home at 22 GB. Pipeline job 35431 started 13:55 (calibration first).
- **Calibration done** (train patient lits_000, 35 liver slices, 45 s with 16 workers):
  trial I0 = 2e5 gave 128.2 HU added noise in the liver; target 48.2 HU ->
  **I0 = 1.41e6 photons per detector element**, written to `configs/lits_sim.json`
  (committed at the time). Conversion job 35431 then went wrong, twice over, both my bugs:
  1. prep_lits.py used the archive's folder as its scratch dir and its cleanup deleted the
     5.2 GB archive after patient 0 -> the other 50 patients "failed" (file not found),
     and the job still exited 0.
  2. the one-step calibration undershot: lits_000 rebuilt at I0 = 1.41e6 had 37.4 HU added
     liver noise, not 48.2. The electronic-noise term (sigma_e = 10 counts) dominates the
     most attenuated rays at low I0, so noise does not scale as 1/sqrt(I0) between 2e5 and 1.4e6.
  Fixes (commit 052ab2c): scratch dir `data/lits_raw_tmp` (refuses to equal the archive
  folder), cleanup removes only .nii files (tested), any failed patient makes the job exit
  1 (so afterok chains stop), patients converted with another I0 are redone, and the
  calibration now ITERATES I0 <- I0 x (std/target)^2 on train patient 0 until within 3 %
  (max 5 rounds, ~1 min each), recording the history in configs/lits_sim.json.
  lits_sim.json was removed from the branch until the new calibration exists.
- RESUBMITTED (14:10): download **35520** -> LiTS build **35521** (afterok) ->
  nnU-Net **35522** (afterok:35521 + afterany:35391). The old nnU-Net job 35434 was
  cancelled (its data dependency had been "satisfied" by the exit-0 failure).

## Step 4: detection metrics (done)

- `scripts/detect_eval.py`: 3D connected components (26-connectivity), lesion detected if any
  prediction voxel overlaps it (optional --min-overlap), detection rate overall and for
  lesions < 10 mm equivalent diameter (sphere of the lesion volume from voxel spacing),
  false positives per patient, per-lesion Dice (in the dilated lesion box) and per-patient
  Dice; works on data/lits slice folders or NIfTI/.npy volumes. 17 synthetic checks in
  tests/check_detect.py pass.

## Step 5: tumour finder (in progress)

- Decision (Q6 default): "nnunet" is a venv on the ctmamba Python with system site
  packages (~/venvs/nnunet; reuses torch). Installed by job 35426 on cpu_student after
  ~20 min of PyPI retries: nnunetv2 2.8.1, 608 MB, imports and CLI verified.
  Trainer nnUNetTrainer_250epochs, 3d_fullres, fold 0 (default 1000 epochs would be ~3-4 days).
- SUBMITTED: nnunet = job 35434 (cancelled later, see step 3) then **35522**, `--dependency=afterok:35521,afterany:35391` (after the
  LiTS build and after the CoreDiff job, so it never runs next to another full-A100 job of
  ours). It writes the raw dataset (train patients only), preprocesses, trains fold 0,
  predicts the locked test patients at normal dose and runs detect_eval.py on them.
- Written: `scripts/prep_nnunet.py` (train patients only -> Dataset501_LiTSLiver; test
  patients images-only for inference, optionally from a model's predictions) and
  `slurm/nnunet_train.sbatch` (raw -> preprocess -> train -> predict normal-dose test
  patients -> detect_eval). Submission waits on data/lits.

## Step 6: Idea 2 hybrid (code + tests done; GPU numbers pending)

- `docs/hybrid_design.md`: CT-Mamba shape map (PFEN -> Haar bands 32x32x32 -> 3 scales of
  Z-scan Mamba per band -> inverse wavelet) and CTformer's T2T dilated attention, plus the
  drop-in block and the 3 variants.
- `ldct/models/ct_mamba_hybrid.py`: variants low / small / parallel (gate sigmoid, init 0.5),
  built from CTformer's own Token_performer + Block; subclasses CTMamba so loss, data,
  schedule stay identical. `configs/models/ct_mamba_hybrid.yaml`, `configs/variants/`.
- `configs/short50.yaml` (cosine over 50 epochs, loss switch at epoch 3, best from 3) and
  `slurm/short50.sbatch` for the 4 chained 50-epoch runs: WRITTEN, NOT SUBMITTED.
- Needed for this: `--run` (named run folders) and multi-file `--config` in train/predict/
  evaluate, and `model_args.warmup_epochs` in the CT-Mamba wrapper (default unchanged).
- tests/check_hybrid.py: CPU part passes; GPU part = job 35429 (MIG 3g.20gb, 19.5 GiB,
  HYBRID_TEST_OK; one process per model, batch 16 where 32 did not fit the slice):
  | model | params | s/step | peak memory | note |
  |---|---|---|---|---|
  | ct_mamba (plain) | 50.8 M | 1.96 @ batch 32 | 18.4 GiB | measured at batch 32 |
  | hybrid low | 59.4 M | 1.09 @ 16 -> ~2.2 @ 32 | 11.9 -> ~24 GiB | estimates scaled x2 |
  | hybrid small | **197.9 M** | 1.47 @ 16 -> ~2.9 @ 32 | 11.3 -> ~23 GiB | see below |
  | hybrid parallel | 61.7 M | 1.16 @ 16 -> ~2.3 @ 32 | 13.0 -> ~26 GiB | gates start at 0.5 |
  All fit a 40 GB A100 at the official batch of 32. **Finding for review:** the "small"
  variant is 4x the size of CT-Mamba because the smallest scale has the WIDEST channels
  (16C = 512 / 1024 / 1024 / 1536 in the four band networks) and the T2T step maps 9C -> C.
  It is a legitimate variant as specified, but not a light one; a channel bottleneck before
  the attention (e.g. 1x1 to 128) would be the obvious fix if you want it. Not changed
  without your decision. Parameter counts exclude the loss network.

## Step 7: Idea 1 lesion loss (code + tests done)

- `ldct/lesion.py`: LesionWeightedL1 (tumour pixels x weight, default 5; exact plain L1
  when no mask), tumour-aware patch sampling (`train.tumour_patch_share`), masks read from
  `<patient>/mask/`. For models with their own loss (CT-Mamba) the weighted L1 is ADDED with
  `lesion_extra_scale` (100 = the scale of CT-Mamba's own L1 term). Not supported for
  models with a custom training_step (CoreDiff) - refused with a message.
- `train.extra_train_sets` (LiTS train part of the split only; the test part is refused),
  `--init <ckpt>` for fine-tuning, LiTS test set in default.yaml now = locked test patients
  from the split file (never "all").
- `configs/finetune_control.yaml` / `finetune_lesion.yaml` + `slurm/finetune.sbatch`:
  WRITTEN, NOT SUBMITTED (also need data/lits first). 24 checks in tests/check_lesion.py pass.

## Checks

- The complete test suite (10 files) passes on the Mac (CPU) and on a DGX compute node
  (job 35433). CT-Mamba per-slice L506 scores regenerated from the existing predictions
  (runs/ct_mamba/scores_mayo_L506.csv, copy in dgx_results/; mean 29.419 dB matches).

## 16 Nov (session resumed after the 12 Nov session was cut off)

What happened on the DGX in between (all from sacct and the logs):
- **CTformer job 35390 COMPLETED** (12 h 22 min on a full A100, 13 Nov 22:58): best.pth =
  epoch 1380 (train loss 0.004357; final-epoch loss 0.004422).
  L506: **PSNR 28.56 dB / SSIM 0.8680 / RMSE 15.28 HU**, 0.112 s/slice.
- **CoreDiff job 35391 COMPLETED** (13 h 54 min, 14 Nov 12:52): best.pth = EMA of epoch 160
  (train loss 1.4e-5). L506: **PSNR 29.28 dB / SSIM 0.8817 / RMSE 14.07 HU**, 0.087 s/slice.
- CT-Mamba stays the best on PSNR/RMSE (29.42 / 13.85); CoreDiff has the best SSIM (0.8817
  vs 0.8776); CTformer is clearly behind. Per-slice scores and full train logs copied to
  dgx_results/ (ctformer_*, corediff_*).
- LiTS build 35521 COMPLETED (2 h 22 min): all **51 patients, 7,274 slices**, 5 patients have
  no tumour; per-patient meta.json copies in dgx_results/lits_meta/. The iterative
  calibration ended at I0 = 9.17e5 (47.5 HU on patient 0). Archive deleted (planned; home
  now 30 GB with the LiTS predictions of two models).
- Because data/lits existed, both baseline jobs also predicted and scored the LiTS test set
  (10 locked patients, 1,458 slices): quarter dose 20.63 dB, CTformer 25.69, CoreDiff 26.36.
  THESE LiTS ROWS ARE PROVISIONAL, see the finding below.

**Finding: one fixed exposure does not give Mayo-like noise across LiTS.** The validation
over all 51 patients (added liver noise per patient): median 70 HU, IQR 55-139, range
32-686 HU (lits_028: 647 HU, lits_032: 211 HU). Mayo's real quarter-dose patients span
37-62 HU. Cause: in the simulation the noise grows with the attenuation path and the pixel
size (corr. with pixel spacing 0.49) and a fixed I0 has no exposure control; real scanners
modulate the tube current per patient (AEC). Decision (logged, reversible): calibrate the
exposure PER PATIENT so that each patient's added liver noise is the Mayo median 48.2 HU
(iterate I0 on up to 12 liver slices until within 3 %, then simulate all slices; the I0
and the iteration history go into each patient's meta.json; configs/lits_sim.json now
records the rule, not a number). Implemented with tests (commit aa1d5dc), and only the
quarter/ slices are rewritten from each patient's own full/ and mask/ slices, so the
pending nnU-Net job (normal dose) is unaffected and no archive is needed.
- SUBMITTED 16 Nov: resimulation **36222** (cpu_student) -> LiTS prediction + evaluation
  on MIG slices for ctformer **36223**, corediff **36224**, ct_mamba **36225** (afterok).
  The quarter-dose LiTS row is recomputed at the end of 36222. The LiTS rows in
  results/summary.csv are replaced by these jobs.
- nnU-Net 35522 still pending (full-A100 queue, Priority).
- SUBMITTED 16 Nov: **detect_all = job 36226** (`slurm/detect_all.sbatch`, MIG slice,
  afterok: 35522 + 36223 + 36224 + 36225). It runs the trained nnU-Net on the locked LiTS
  test patients as normal dose, quarter dose and each model's denoised output, scores
  lesion detection with detect_eval.py and writes results/detect_<input>.json and
  results/detect_summary.csv. prep_nnunet.py gained `--from-kind quarter`;
  tests/check_nnunet_prep.py (7 checks) added. This is the whole evaluation chain the
  project is about, fully queued; it needs no further input unless a job fails.
- Job 36222 FAILED after 2 min (0 patients): the `--from-existing` branch of prep_lits.py had
  not been inserted inside the patient loop (a text replacement missed silently) and the
  job exited 1 as designed, so the dependants 36223-36226 went "DependencyNeverSatisfied".
  Fixed with a loop-level test. RESUBMITTED: resim **36228** -> LiTS
  predict/evaluate ctformer **36229**, corediff **36230**, ct_mamba **36231** -> detect_all
  **36232** (also after nnU-Net 35522). 36223-36226 cancelled.
- 16 Nov, +1h30: 36228 running on dgxb, 26/51 patients resimulated, every one within
  47.6-48.8 HU of the 48.2 HU target (3-5 calibration rounds, ~4 min per patient,
  per-patient I0 from 4.5e5 to 1.4e7 photons). ~1.5 h left; 36229-36232 wait on it.
- 16 Nov: **36228 COMPLETED** (2 h 49 min): 51/51 patients resimulated, 0 failed, home 29.8 GB.
  Per-patient added liver noise now 45.9-51.9 HU, median 48.5 (calibrate_lits_noise.py,
  Mayo scorer; the in-job liver-mask ROI gave 44.1-49.5). Exposure I0 ranges 4.5e5-1.4e7
  photons (median 1.66e6), 1-6 calibration rounds each. Test patients: 46.6-51.9 HU.
  Quarter-dose LiTS baseline recomputed: 24.29 dB / 0.8193 / 25.77 HU on 1,458 slices.
  Copied back: configs/lits_sim.json, dgx_results/lits_added_noise.json,
  dgx_results/lits_meta/*/meta.json (51), dgx_results/summary.csv (model LiTS rows in it
  are still the OLD simulation until 36229-36231 overwrite them).
  Queue: 36229 ctformer RUNNING on dgxa; 36230/36231 pending (AssocGrpGRES = MIG group
  limit, they start when a slot frees); 36232 detect_all on dependency; 35522 nnU-Net
  still Priority-pending in the full-A100 queue.
- 16 Nov: **36229/36230/36231 COMPLETED** (7 min 55 s / 6 min 21 s / 58 min 13 s, no errors).
  LiTS locked test set, 1,458 slices: CTformer 28.51 dB / 0.8354 / 15.71 HU; CoreDiff
  29.43 / 0.8560 / 14.08; CT-Mamba 29.92 / 0.8608 / 13.53; quarter dose 24.29 / 0.8193 /
  25.77. Same ranking as Mayo L506. Copied back dgx_results/summary.csv (final) and
  dgx_results/<model>_scores_lits.csv (per slice). Home 32 GB.
  Remaining queue: nnU-Net 35522 (Priority, full-A100 queue) -> detect_all 36232.
- 16 Nov, evening: nnU-Net 35522 still Priority-pending; `squeue --start` estimates
  2026-11-20 10:36. Five full-A100 jobs from other users are running (longest 5 d 5 h).
  Nothing in the plan depends on it except detect_all 36232, so the chain just waits.
- 17 Nov: **nnU-Net 35522 COMPLETED** (started 16:10, 4 h 11 min, all 250 epochs, 32 train /
  9 validation cases). Final epoch: train loss -0.755, val loss -0.674, pseudo Dice liver
  0.965 / tumour 0.681, ~50 s per epoch. Its predictions on the locked test patients
  (normal dose) are in runs/nnunet/pred_test_full.
- **detect_all 36232 FAILED** after 7 min, and the scoring step inside 35522 hit the same
  error: detect_eval.py read nnU-Net's NIfTI predictions as (x, y, z) but the ground truth
  slice folders as (z, y, x) ("prediction (512, 512, 149) and ground truth (149, 512, 512)
  differ in shape"). Fixed: load_volume now transposes NIfTI back to (z, y, x), matching
  prep_nnunet.save_nifti. New round-trip test in tests/check_detect.py; detection and
  nnU-Net prep tests pass. Fix copied to the DGX. RESUBMITTED as **detect_all 38156**.
  Home 39 GB (each detection pass adds ~0.5 GB temporarily and deletes it).
- 18 Nov: **detect_all 38156 COMPLETED** (40 min). Same nnU-Net on the 10 locked LiTS test
  patients, 34 tumours (9 under 10 mm):
  | input | detected | small detected | FP / patient | per-patient Dice | per-tumour Dice |
  |---|---|---|---|---|---|
  | normal dose | 27/34 | 5/9 | 0.6 | 0.605 | 0.582 |
  | quarter dose | 28/34 | 5/9 | 0.5 | 0.601 | 0.565 |
  | CT-Mamba | 27/34 | 5/9 | 0.3 | 0.576 | 0.543 |
  | CoreDiff | 28/34 | 5/9 | 0.5 | 0.594 | 0.584 |
  | CTformer | 28/34 | 5/9 | 0.5 | 0.599 | 0.585 |
  Detection counts are saturated: 5 tumours (2.2-7.1 mm and one 20.5 mm) are missed on every
  input, two more flip by chance. Paired per-tumour Dice (new scripts/detect_compare.py ->
  dgx_results/detect_paired.json, Wilcoxon): quarter vs normal -0.017 (p = 0.018, the only
  significant difference); CoreDiff +0.003 and CTformer +0.003 vs normal (recover the loss);
  CT-Mamba -0.039 vs normal (p = 0.16), lowest on small tumours too. Best PSNR is not best
  tumour outline: supports Idea 1. Suggested next: lower-dose test (10 %, 5 %) and more
  tumours (LiTS part 2) to make detection sensitive.
- 18 Nov: professor report built: docs/report/LDCT_Project_Report.pdf (8 pages, all numbers
  read from dgx_results by docs/report/make_report.py).

# Final week (branch `final-week`, from `autonomous-nov12`), started 18/19 Nov 2026

Owner: "you make a plan", full access, good progress by the morning of 19 Nov. Defaults taken
for all open questions (logged here): equal training PIXELS for refiner and control; success
criterion (b) "best baseline" = best of CT-Mamba/CTformer/CoreDiff; significance = Wilcoxon
p < 0.05 AND bootstrap 95% CI excluding 0; hybrid 50-epoch runs dropped (out of scope).

- **Step 1 DONE**: docs/success_criteria.md committed (4415a52) before any refiner/control result.
- **DEADLINE RISK 1, GPU: the account may run only ONE GPU job at a time** (association
  GrpTRES gres/gpu=1, any MIG size; this is what "AssocGrpGRES" meant on 16 Nov). So refiner and
  control CANNOT run in parallel; everything GPU is a single queue: control -> precompute ->
  refiner -> predictions -> detection -> extended set. All 7 full A100s were busy with 39
  full-A100 jobs pending, so the control runs on a free a100_4g.20gb MIG slice.
- **DEADLINE RISK 2, disk**: home 40.96 GB (du -sb) of 50; the 10-GB-free rule leaves ~1 GB, and
  the Stage-A precompute needs ~5 GB (int16, 5,410 Mayo + 5,816 LiTS train slices). The plan
  is to free ~8 GB by archiving data/nnunet/raw (+ preprocessed, rebuildable), the CTformer
  and CoreDiff LiTS predictions and runs/ct_mamba/latest.pth to the owner's laptop and then
  delete them on the DGX. The archive copy is running from the laptop; the delete itself was
  refused by this session's safety check, so it needs the owner's explicit OK (or the owner
  runs it). Until then the precompute and the refiner cannot start; the control can.
- LiTS part 2: per-file `kaggle -f` answers 404 again (18 Nov), and the archives (11.6 GB part
  2 + 5.2 GB part 1 for the masks) do not fit. NEW scripts/remote_zip.py reads single members
  straight from the remote archive with HTTP range requests (zip CRC checked), so no archive
  is stored: 39 MB mask in 25 s.
- **Step 6 started**: job **39373** (cpu_student) scans the masks of patients 51-130 (tumour
  counts / sizes counted exactly like detect_eval, no images or models) ->
  results/lits_ext_scan.json, to choose the extended test set before anything is scored.
- **Step 3/4 code DONE** (b82f0e0): ldct/tumour_train.py (MixedPatches: deterministic patch i =
  f(seed, i); even i Mayo, odd i LiTS train; 30% of LiTS patches centred on tumour; l1_lesion =
  mean|e| + 4 x mean(mask x |e|), masks dilated 3 px), ldct/models/ct_mamba_refiner.py
  (2D U-Net 32/64/128/256, GroupNorm + SiLU, bilinear up, zero-init head: 2,143,169
  parameters), scripts/train_tumour.py (iteration-based, AdamW 2e-4 cosine, wd 1e-4, best.pth =
  lowest 1000-iteration window loss), tests/check_refiner.py (26 checks, all pass on the Mac and
  on the DGX).
- Pixel matching (logged): refiner 16 x 128^2 = control 64 x 64^2 = 262,144 pixels per step,
  30,000 steps each (7.9e9 pixels). CT-Mamba takes only 64x64 inputs. 64 patches did not fit in
  20 GB (job 39376 OOM), so the control accumulates gradients over 2 x 32 patches: the same
  262,144 pixels per optimizer step. The control trains CT-Mamba's denoiser only (its
  loss-side noise nets are not used by this loss); 50.8 M trainable parameters.
- best.pth and best_from11.pth of CT-Mamba hold identical weights (epoch 237), so the existing
  CT-Mamba predictions (runs/ct_mamba/predictions/{mayo_L506,lits}) are valid Stage-A inputs
  for testing the refiner.
- **Step 4 SUBMITTED**: control = job **39377** (a100_4g.20gb).
- Control 39377: first window 1,000 iterations in 1,306 s -> 30,000 in ~10.9 h (ends ~11:20
  19 Nov); 31% of LiTS patches contain tumour.
- **Step 6 mask scan 39373 DONE** (31 min): patients 51-130, 626 tumours (330 < 10 mm), 72
  patients with tumour, 13,199 liver slices (results/lits_ext_scan.json).
- **Extended test split COMMITTED before any conversion or scoring** (configs/lits_ext_split.json):
  10 new patients 51 58 63 71 79 88 92 112 118 120 with 72 tumours (49 < 10 mm), 1,156 slices.
  Extended test set = existing 10 + these = 20 patients, 106 tumours, 58 < 10 mm.
  Rule revision (logged, before any conversion/scoring): the first rule (96ae71f, no tumour cap)
  picked ONE patient, lits_117 with 75 tumours, which alone reached 100; a set dominated by one
  patient makes per-tumour statistics and FP/patient meaningless, so patients with > 10 tumours
  are now excluded (7 such patients, incl. 117 and 108 with 66).
- Conversion (slurm/lits_ext_prep.sbatch, ~1.5 GB) NOT submitted yet: home is at ~39 GiB, the
  conversion would pass the keep-10-GB-free line. Waiting for the disk decision.

## 19-20 Nov
- **Control 39377 COMPLETED** (11 h 3 min, 30,000 iterations, best window loss 0.004045).
  The chained jobs stopped cleanly at the disk guards as designed: eval_control 39378 (DISK_STOP,
  home 39 GB) and stage_a 39379 (stopped at 1,000/11,226 slices, home 39.9 > 39.7 GB);
  refiner 39380 / eval_refiner 39381 then left the queue.
- **Disk solution without deleting anything:** the compute nodes have 237 GB free on node-local
  /tmp (outside the home quota). New `slurm/refiner_pipeline.sbatch` runs control evaluation,
  Stage A, refiner training and evaluation in ONE GPU job and keeps every large temporary file
  (Stage A ~6 GB, predictions ~2 GB per model, nnU-Net input copies) in /tmp/ldct_<user>_<job>,
  removed on exit. Supporting changes: `LDCT_STAGE_A_ROOT` in ldct/tumour_train.stage_a_path,
  free-space guard on the scratch in precompute_stage_a.py, `SETNAME`/`NNWORK` in detect_all.sbatch.
- **Owner decision (20 Nov, before any refiner result): add a 2.5D refiner variant**
  (`refiner25d`): same U-Net, same loss, data, patches (identical sampler draws) and pixel
  budget, but input [noisy z-1, noisy z, noisy z+1, Stage A] (in_channels 4). Reason: adjacent
  slices add real information (independent noise, shared anatomy) that should raise PSNR;
  CoreDiff, which uses context slices, has the best SSIM. The planned 2D refiner stays the main,
  pre-committed experiment; the 2.5D model is reported as a second variant against the same
  criteria. Concern discussed with the owner: the refiner trains on 128 patches but runs on
  512 slices and GroupNorm statistics depend on image size; if its PSNR drops, tiled inference
  is the first fix (inference only).
- SUBMITTED 20 Nov: **refiner_pipeline = job 42024** (control eval -> Stage A -> refiner -> refiner eval -> refiner25d -> refiner25d eval). tests/check_refiner.py (36 checks incl. 2.5D + scratch path) ALL PASSED on a DGX node.
- **Job 42024 COMPLETED 20 Nov 17:25** (7 h 9 min, 0:0). Home stayed at 39.9-40 GB the whole time
  (all temporaries on node /tmp, removed). Live check: stored Stage A = live CT-Mamba to
  <= 0.026 HU on 3 L506 + 3 LiTS slices. Training: refiner best window loss 0.003978,
  refiner25d 0.003826 (30,000 iterations each, ~31 s per 1,000-iteration window, ~16 min per model).

### First evaluation (existing 10 LiTS patients, 34 tumours, 9 < 10 mm; L506 526 slices) - PRELIMINARY
`scripts/final_stats.py --refiner {ct_mamba_refiner|ct_mamba_refiner25d}` ->
dgx_results/final_existing10{,_25d}.{md,json,_image.csv,_detection.csv,_paired.csv}

| input | L506 PSNR [95% CI] | SSIM | detected | < 10 mm | FP/pt | per-tumour Dice |
|---|---|---|---|---|---|---|
| CT-Mamba | 29.42 [29.26, 29.57] | 0.8776 | 27/34 | 5/9 | 0.30 | 0.543 |
| CTformer | 28.56 | 0.8680 | 28/34 | 5/9 | 0.50 | 0.585 |
| CoreDiff | 29.28 | 0.8817 | 28/34 | 5/9 | 0.50 | 0.584 |
| control | 29.41 [29.26, 29.57] | 0.8775 | 28/34 | 5/9 | 0.70 | 0.582 |
| refiner (2D) | 29.42 [29.26, 29.57] | 0.8775 | 28/34 | 5/9 | 0.50 | 0.582 |
| refiner 2.5D | **29.64** [29.48, 29.79] | **0.8826** | 28/34 | 5/9 | 0.60 | 0.585 |

Criteria (docs/success_criteria.md):
- 2D refiner: a FAIL (29.4176 < 29.42; -0.001 dB vs CT-Mamba, i.e. the residual learned almost
  nothing on Mayo), b FAIL (0.5819 < CTformer 0.5852), c FAIL (0.5819 vs control 0.5820),
  d FAIL, e FAIL (0.5 > 0.3). **NOT A WIN.**
- 2.5D refiner (second variant, decided 20 Nov before results): a PASS (+0.22 dB vs CT-Mamba,
  paired CI [+0.215, +0.222], p < 1e-4; also +0.36 dB vs CoreDiff, best SSIM of all inputs),
  b FAIL by 0.0006 (0.5846 vs 0.5852), c PASS on the point estimate (+0.0026) but not
  significant (CI [-0.0030, +0.0085], p 0.52), d FAIL, e FAIL (0.6). **NOT A WIN.**
- Detection counts are identical for every denoised input (28/34, small 5/9); denoising does not
  change which tumours the nnU-Net finds on this set. Dice differences between all trained
  denoisers except CT-Mamba are <= 0.004 and not significant; CT-Mamba alone is lower (refiner,
  2.5D and control all beat it, p 0.02 / 0.02 / 0.06).
- Caveat: two variants are judged against the same criteria, so a single pass would need a
  multiple-comparison note; not relevant since neither passes. The extended 20-patient set
  (step 6) is the final verdict.

## 21 Nov
- Step 6 rewritten as ONE GPU job on node-local scratch: `slurm/ext_pipeline.sbatch` (conversion
  of the 10 extended patients from the remote Kaggle archives -> $SCR/lits_ext, quarter-dose
  scores, nnU-Net detection on normal/quarter dose, then CT-Mamba, refiner, refiner25d, control,
  CTformer, CoreDiff: predict -> score -> detect -> delete). `LDCT_LITS_EXT_ROOT` (ldct/config.py)
  points the lits_ext test set at the scratch; the prep disk guard measures the scratch.
  Patient meta.json files are copied to results/lits_ext_meta. Replaces lits_ext_prep + eval_ext.
- SUBMITTED 21 Nov: **ext_pipeline = job 42584**.
- Job 42584 waited ~4 h for a 4g slice; measured peak GPU memory on a 2g.10gb slice (job 42640):
  CT-Mamba 1.7 GiB (3.8 s/slice), CTformer 1.3, CoreDiff 0.7, refiners 0.4 GiB, so the job was
  moved to 2g.10gb and started.
- **BUG FOUND AND FIXED (21 Nov): left-right flipped LiTS masks.** The masks of LiTS volumes
  48-130 carry an identity NIfTI header (RAS, 1 mm) while their scans are LAS with real spacing;
  the voxel arrays are aligned as stored, but `load_canonical` reoriented each file by its own
  header, which flipped the mask left-right relative to the scan. Shown on lits_051 (raw files):
  mean HU inside the liver mask 79.8 as stored vs -83.5 flipped. Job 42584 was cancelled after
  4 min, before any extended-set data was used. Fix in scripts/make_lits_lowdose.py: the mask
  takes the scan's affine when the headers disagree (`mask_affine`, logged as NOTE), and every
  conversion prints a liver-mask alignment check. Re-tested on lits_051 (job 42646): 79.8 HU
  (flipped -83.5); stored slices 84.4 vs -57.1 flipped.
- **Impact on existing data (aligncheck over all 51 part-1 patients, job 42645):**
  all 10 locked test patients are aligned (first evaluation unaffected). Training patients
  lits_048, lits_049, lits_050 (the only part-1 patients with the header mismatch) have
  left-right flipped masks: liver-mask mean 34 / 41 / 38 HU vs 95 / 109 / 124 HU flipped. These
  3 of 41 LiTS training patients therefore had wrong tumour labels for (i) the evaluation
  nnU-Net's training and (ii) the tumour-weighted loss and tumour-centred patches of refiner,
  refiner25d and control. Not retrained (deadline; the nnU-Net is the same fixed evaluator for
  every input); reported as a limitation.
- The extended-set selection (scan_lits_ext.py) measured tumour diameters with the masks' 1 mm
  header spacing instead of the scans' spacing (e.g. 0.90 x 0.90 x 3 mm), so diameters were
  underestimated by about the cube root of the voxel volume (~1.3x) and the "58 < 10 mm" count
  is too high. The selected patients stay as committed (changing them after the fact would be a
  post-hoc choice); the evaluation (detect_eval) uses the scans' spacing from meta.json, so the
  reported small-tumour counts will be the correct ones.
- RESUBMITTED: **ext_pipeline = job 42648** (2g.10gb).
- **ext_pipeline 42648 COMPLETED 21 Nov 21:06** (~5 h on 2g.10gb). All 10 conversions passed the
  alignment check (liver-mask mean 80-139 HU, flipped -137..+55); only lits_051 needed the header
  fix. Live Stage-A check 0.0 HU. Correction to the note above: only lits_051 of the extended
  patients had the identity mask header, so the selection's small-tumour count was off by one:
  the evaluation counts **57** tumours < 10 mm of 106 (not 58).

### FINAL evaluation: 20 LiTS patients, 106 tumours (57 < 10 mm) + Mayo L506
`scripts/merge_detect.py` (10 + 10 patients) -> `final_stats.py --detect-results dgx_results/combined20`
-> dgx_results/final_extended20{,_25d}.*

| input | detected | < 10 mm | FP/pt | per-tumour Dice [95% CI] | Dice < 10 mm | LiTS-ext PSNR |
|---|---|---|---|---|---|---|
| normal dose | 70/106 | 26/57 | 0.40 | 0.417 [0.348, 0.486] | 0.220 | - |
| quarter dose | 60/106 | 18/57 | 0.35 | 0.365 [0.296, 0.438] | 0.170 | 23.91 |
| CT-Mamba | 64/106 | 20/57 | 0.30 | 0.354 [0.288, 0.421] | 0.175 | 30.54 |
| CTformer | 68/106 | 23/57 | 0.40 | 0.416 [0.346, 0.485] | 0.222 | 29.02 |
| CoreDiff | 64/106 | 19/57 | 0.40 | 0.378 [0.311, 0.448] | 0.193 | 29.79 |
| control | **71/106** | **26/57** | 0.65 | **0.417** [0.350, 0.484] | **0.231** | 31.62 |
| refiner 2D | 69/106 | 24/57 | 0.50 | 0.399 [0.330, 0.469] | 0.201 | 31.40 |
| refiner 2.5D | 68/106 | 23/57 | 0.50 | 0.371 [0.302, 0.441] | 0.173 | **31.90** |

Criteria: refiner 2D FAIL a b c d e; refiner 2.5D PASS a, FAIL b c d e. **Final verdict: NOT A WIN
for either.** Paired per-tumour Dice: refiner - control -0.018 [-0.029, -0.008] p 0.0007
(significantly WORSE); refiner - CT-Mamba +0.045 p < 1e-4; refiner - CTformer -0.017 p 0.17;
2.5D - control -0.046 p < 1e-4; 2.5D - CTformer -0.044 p 0.002; 2.5D - normal dose -0.046 p 0.001.
Control - CT-Mamba +0.063 p < 1e-4.

Reading:
1. The extra training on LiTS with the tumour-weighted loss is what helps: the fine-tuned
   control reaches normal-dose Dice (0.417 vs 0.417) and normal-dose small-tumour detection
   (26/57), up from CT-Mamba 0.354 and 20/57; it costs false positives (0.65 vs 0.30).
2. The refiner design does not add to that; the 2D refiner is significantly below the control.
3. The 2.5D refiner has the best image scores (L506 29.64 dB, LiTS-ext 31.90 dB) and the
   second-worst trained-model Dice: on 106 tumours, higher PSNR again does not mean better
   tumour outlines - the project's central finding, now on a larger set.
4. The first 10-patient result (all models equal) was too small to separate them; on the
   new patients the differences are large and significant.
