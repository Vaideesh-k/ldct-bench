# Morning report, session of 12 Nov 2026

Everything is on branch **`autonomous-nov12`** (pushed; please review and merge). The
full log with decisions is docs/PROGRESS.md. Steps 1, 2, 4, 6 and 7 of your plan are
done; 3 and 5 are done up to the point where cluster jobs take over, and those jobs are
submitted and chained.

## What finished

| step | result |
|---|---|
| 1 Sync | slurm/ scripts recovered from the DGX and committed; DGX code current; 0.9 GB freed |
| 2 Baselines | CTformer and CoreDiff wrapped from the pinned official repos with their official settings (differences documented in the wrapper docstrings); smoke tests passed on a MIG slice; full runs submitted and chained |
| 3 LiTS | locked split committed (41 train / 10 test of part 1); simulator + tests; Mayo noise target measured (48.2 HU); dose calibrated (I0 = 1.41e6, `configs/lits_sim.json`); conversion of all 51 patients running |
| 4 Detection metrics | `scripts/detect_eval.py` + 17 tests |
| 5 Tumour finder | nnU-Net v2 2.8.1 installed in the `nnunet` venv; dataset script + job written; job submitted behind the data and GPU chain |
| 6 Idea 2 hybrid | `docs/hybrid_design.md`, `ldct/models/ct_mamba_hybrid.py` (3 variants), tests incl. GPU measurements, 50-epoch job scripts written, NOT submitted |
| 7 Idea 1 lesion loss | `ldct/lesion.py` (weighted L1 + tumour-aware sampling), extra train sets, `--init`, fine-tune configs + job script written, NOT submitted |
| extras | per-slice CT-Mamba L506 scores (dgx_results/), README updated, all 10 test files pass on Mac and DGX |

## What is running / queued on the DGX

| job | what | state when I last looked | expected |
|---|---|---|---|
| 35520 | re-download of the LiTS part-1 archive (my first pipeline run deleted it, see Problems) | pending (cpu_student) | ~25 min |
| 35521 | LiTS build: iterative dose calibration on train patient 0, then all 51 patients, test patients first | pending, after 35520 | ~1-2 h; stops by itself above 38 GB used; exits 1 if any patient fails |
| 35390 | CTformer full run (530k updates) + predict + evaluate L506 | pending, 21st in the full-A100 queue | SLURM estimate: starts 14 Nov 00:23, then ~1 day |
| 35391 | CoreDiff full run (150k updates) + predict + evaluate L506 | pending, after 35390 | ~1.5 days after CTformer |
| 35522 | nnU-Net 3d_fullres, 250-epoch trainer, fold 0, then predict the locked test patients at normal dose + detect_eval | pending, after 35521 and 35391 | ~0.5-1 day once it starts |

Results land in results/summary.csv, runs/<model>/scores_mayo_L506.csv and
results/detect_nnunet_fulldose.json on the DGX; logs in ~/ldct-bench/logs/.

## Results so far

- Mayo quarter-dose added noise in uniform liver-like tissue: median 48.2 HU (IQR 41-55),
  full-dose std there 23.7 HU.
- LiTS simulation: trial at I0 = 2e5 gave 128 HU added liver noise, the one-step estimate
  1.41e6 gave 37.4 HU (electronic noise breaks the 1/sqrt law), so the calibration now
  iterates; the final I0 will be in configs/lits_sim.json on the DGX (copy it to the repo:
  `scp dgx:ldct-bench/configs/lits_sim.json configs/` and commit) and in logs/lits_prep_35521.out.
- Hybrid measurements (MIG slice, batch 32, estimates for the hybrids): CT-Mamba 50.8 M /
  1.96 s/step / 18.4 GiB; low 59.4 M / ~2.2 / ~24 GiB; small 197.9 M / ~2.9 / ~23 GiB;
  parallel 61.7 M / ~2.3 / ~26 GiB. All fit a 40 GB A100.
- Baseline timings: CTformer ~1 day, CoreDiff ~1.4 days at official settings.

## Problems

- Kaggle refuses per-file downloads for this dataset (404 everywhere), so the whole 5.2 GB
  part-1 archive is downloaded on a CPU node and patients are extracted one at a time.
- My first LiTS pipeline run deleted that archive after patient 0 (scratch folder = archive
  folder) and reported success; fixed and tested (commit 052ab2c), archive re-downloading
  as job 35520. Cost: ~1 h and 5 GB of traffic. Delete data/lits_raw/*.zip after 35521 is done.
- The full-A100 queue is long: CTformer will wait about 1.5 days before it starts.
- Hybrid variant "small" is 4x the size of CT-Mamba (the 8x8 scale has the widest channels);
  see docs/hybrid_design.md for the one-line fix if you want a lighter version.

## Decisions I made on my own

- Official baseline schedules run in full (both under the 5-day cap); shared cosine
  schedule and best.pth rule applied to both; every difference documented.
- CoreDiff keeps the first/last slices (edge neighbour repeated) so all 526 L506 slices score.
- Calibration patient = a train patient (lits_000), never a test patient.
- Hybrid parallel gate initialised to 0.5; attention block = CTformer's Token_performer +
  Block with dropout 0.1, 8 heads.
- Idea 1 extra L1 term scaled x100 for CT-Mamba (same scale as its own L1 term).
- nnU-Net: 250-epoch trainer, 3d_fullres, venv instead of a conda env.
- The LiTS test set in configs now reads the locked split (never "all patients").

## What I need from you

1. Review and merge `autonomous-nov12` (or tell me what to change).
2. Idea 2: approve or change the three variants (especially "small"), then submit the four
   50-epoch runs with the commands in the header of slurm/short50.sbatch (~14 h each,
   chained; needs a free full A100 slot after the baselines).
3. Idea 1: once data/lits exists, submit slurm/finetune.sbatch (control first, then lesion;
   commands in its header).
4. Decide whether to move CTformer to a MIG slice to skip the queue (it needs 3.3 GiB):
   `scancel 35390 35391` then resubmit with `--gres=gpu:a100_3g.20gb:1` and re-chain CoreDiff.
5. After the nnU-Net job: run the denoisers on the low-dose LiTS test patients
   (`predict.py --sets lits`), then nnU-Net on each model's output and detect_eval.py;
   that is the detection comparison the project is about. The scripts exist; it needs the
   trained models first.
