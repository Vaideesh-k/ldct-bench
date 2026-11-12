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
- **SUBMITTED: nnunet = job 35434**, `--dependency=afterok:35431,afterany:35391` (after the
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
