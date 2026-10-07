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
    data.py              HU scaling, slice pairs, patches, batching
  scripts/               runnable scripts
    prep_mayo.py         Mayo DICOM -> data/mayo/<patient>/{quarter,full}/*.npy
  tests/                 quick checks using small fake data (no real CT needed)
  requirements.txt       Python packages, pinned versions
```

Not committed (see `.gitignore`): `data/`, `runs/`, `third_party/`,
checkpoints (`*.pth`) and arrays (`*.npy`).

## Setup

```bash
git clone https://github.com/Vaideesh-k/ldct-bench.git
cd ldct-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check everything works (uses fake data, takes a few seconds):

```bash
python tests/check_data.py
python tests/check_prep.py
```

## Preparing the Mayo data

CT data is not in git. Get the Mayo 2016 DICOM folders (L067, L096, ..., L506),
then convert them:

```bash
python scripts/prep_mayo.py --src /path/to/Mayo2016 --dst data/mayo
```

The script stops with an error if a patient's quarter-dose and full-dose slices
do not line up.
