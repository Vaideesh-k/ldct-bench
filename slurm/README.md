# SLURM job scripts for the university DGX A100

All heavy work runs through SLURM from `~/ldct-bench` on the DGX. Paths inside the
scripts are absolute (`/dgxa_home/se23uari128/ldct-bench`), so change them if you
run them from another account.

| File | What |
|---|---|
| `setup_env.sh`, `setup_env2.sh` | one-time creation of the `ctmamba` conda env (run on a compute node; PyPI is flaky there, see setup_env2.sh) |
| `verify_env.py` | import + tiny Mamba forward/backward check of that env |
| `ctmamba_smoke.sbatch` | CT-Mamba smoke test + step timing (`bench_step.py`) on one A100 |
| `ctmamba_train.sbatch` | the 250-epoch CT-Mamba run (job 27804): `run_all.sh ct_mamba` |
| `watch_best.sbatch`, `watch_best.py` | CPU-side watcher that kept `best_from11.pth` before `train.best_from_epoch` existed in `train.py` (now redundant, kept for the record) |

Rules: one long full-A100 job at a time (chain with `--dependency=afterany:<jobid>`),
short tests on MIG slices (`--gres=gpu:a100_3g.20gb:1`), CPU work on `cpu_student`,
nothing heavy on the login node.
