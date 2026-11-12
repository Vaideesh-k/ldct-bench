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
