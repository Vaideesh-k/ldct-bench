# Morning report — final week, night of 18/19 Nov 2026

Branch **`final-week`** (from `autonomous-nov12`, pushed after every step). Full log: docs/PROGRESS.md.

## Done tonight
| step | status |
|---|---|
| 1 success criteria | **DONE**, committed before any result (docs/success_criteria.md, 4415a52) |
| 3 refiner code + tests | **DONE**: `ct_mamba_refiner` (2D U-Net 32/64/128/256, zero-init residual on CT-Mamba, 2.14 M parameters), shared sampler/loss, 26 checks pass (Mac + DGX) |
| 4 control | **RUNNING**, job 39377 on a 4g.20gb MIG slice, 30,000 iterations at ~1.35 s → ~11 h |
| 2 precompute | written, **queued** (39379); needs ~5 GB of disk, see "Blocker" |
| 3 refiner training | **queued** (39380, after the precompute) |
| 5 first evaluation | jobs queued (39378 control, 39381 refiner): L506 + LiTS images, nnU-Net detection; then `scripts/final_stats.py` (bootstrap CIs, paired tests, criteria verdict) |
| 6 extended test set | mask scan of patients 51-130 **running/done** (39373); selection rule committed before conversion (scripts/make_lits_ext_split.py); conversion + evaluation jobs written |
| 7 report material | figure + training-curve script written (scripts/make_figures.py); tables come from final_stats.py |

## Two things that threaten the deadline
1. **Only ONE GPU job at a time on this account** (any MIG size; association limit gres/gpu=1).
   Refiner and control cannot run in parallel, so everything GPU runs in one chain:
   control (~11 h) → control eval (~1.5 h) → precompute (~5 h) → refiner (~2 h) → refiner eval
   → extended-set eval (~5 h). First evaluation expected about 20 Nov morning if the 4g slice
   keeps being free between jobs. Full A100s were all busy (39 jobs pending).
2. **Disk.** Home is at 39 of 50 GB; the keep-10-GB-free rule leaves ~1 GB, and the next jobs need
   ~5 GB (precompute) + ~2 GB (each new model's test predictions) + ~3 GB (extended set). Every
   job checks the disk first and stops cleanly instead of breaking the rule.

## What I need from you (one line answers are enough)
- **OK to delete these on the DGX** once the laptop copy has them (I compare sizes first):
  `data/nnunet/raw`, `data/nnunet/preprocessed` (both rebuildable from data/lits),
  `runs/ctformer/predictions/lits`, `runs/corediff/predictions/lits` (detection already scored),
  `runs/ct_mamba/latest.pth` (resume file of a finished run). That frees ~8 GB. My session's
  safety check refused the delete without your explicit OK. Reply "ok delete" and I do it.
- Keep the laptop copy running until it finishes (it resumes on its own).
