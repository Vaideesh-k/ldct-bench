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
