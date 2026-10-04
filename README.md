# ldct-bench

A shared pipeline to compare low-dose CT denoising models
(CT-Mamba, CTformer, CoreDiff) on the same data, splits and settings.
Each teammate adds only one model file; everything else is shared.

**Locked test patient:** Mayo patient `L506` is used only for final testing.
It must never appear in `train_patients` or `val_patients`.

## Folder map

```
ldct-bench/
  configs/default.yaml   shared settings (data splits, training, inference)
  configs/models/        one yaml per model, overriding default.yaml
  ldct/                  the Python package
    config.py            loads settings (load_config, resolve, ROOT)
    models/              model registry; one file per model
  scripts/               runnable scripts (train, test, ...)
```

Not committed (see `.gitignore`): `data/`, `runs/`, `third_party/`,
checkpoints (`*.pth`) and arrays (`*.npy`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml
```
