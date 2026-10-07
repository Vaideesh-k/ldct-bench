# ldct-bench

A shared pipeline to compare low-dose CT denoising models (CT-Mamba, CTformer,
CoreDiff, ...) on exactly the same data, splits, training rules and metrics.
Each teammate adds **one model file**; everything else is shared, so the
numbers in `results/summary.csv` can be compared fairly.

## Fixed rules (do not change)

| Rule | Where it is enforced |
|---|---|
| Train on Mayo 2016, 1 mm: L067 L096 L109 L143 L192 L286 L291 L310 L333 | `configs/default.yaml` |
| **L506 is the locked test patient**: never in training or validation | `ldct/config.py` refuses to load such a config; `train.py` refuses to run |
| Final model = epoch with the **lowest training loss** (`best.pth`), as in CT-Mamba | `scripts/train.py` |
| Validation (every 20th slice of L333) only logs a loss; it never picks the model | `scripts/train.py` |
| Slices on disk: `.npy` in HU (int16 or float32). Models see [0, 1] via `hu_to_unit` | `ldct/data.py` |
| Metrics in the soft-tissue window -160..240 HU | `ldct/metrics.py`, `eval.window` |

## Folder map

```
ldct-bench/
  configs/default.yaml        shared settings (splits, training, inference, metrics window)
  configs/models/<name>.yaml  one per model: only what differs from default.yaml
  ldct/                       the Python package
    config.py                 load_config(), resolve(), the L506 guard
    data.py                   HU scaling, slice pairs, training patches, test sets
    infer.py                  denoise a whole slice (directly, or in overlapping tiles)
    metrics.py                PSNR / SSIM / RMSE in the HU window
    runtime.py                pick cuda/mps/cpu, honest GPU timing
    models/__init__.py        build_model(name): the registry
    models/template.py        copy this to add a model (documents the contract)
    models/redcnn.py          RED-CNN, small reference model for testing
    models/ct_mamba.py        CT-Mamba, wraps the official code in third_party/
  scripts/
    prep_mayo.py              Mayo DICOM -> data/mayo/<patient>/{quarter,full}/*.npy
    rearrange_mayo.sh         fix the folder layout of the unzipped Mayo files
    get_ct_mamba.sh           download the official CT-Mamba code (pinned commit)
    train.py                  train one model -> runs/<model>/best.pth
    predict.py                denoise all test slices + timing
    evaluate.py               score them -> results/summary.csv
    run_all.sh                train (resumable) + predict + evaluate in one go
  tests/                      quick checks on small fake data (no real CT needed)
  results/summary.csv         the comparison table (committed)
  LAPTOP_RUN.md               step-by-step guide for the GPU laptop
```

Not in git: `data/` (CT data), `runs/` (checkpoints, predictions), `third_party/`
(official model code), `*.pth`, `*.npy`.

## Setup

```bash
git clone https://github.com/Vaideesh-k/ldct-bench.git
cd ldct-bench
python3 -m venv .venv && source .venv/bin/activate   # or: conda activate ctmamba
pip install torch                                      # only if your environment has no PyTorch yet
pip install -r requirements.txt
```

Check everything works (fake data, a minute or two):

```bash
python tests/check_data.py
python tests/check_prep.py
python tests/check_metrics.py
python tests/check_infer.py
python tests/check_models.py
python tests/check_pipeline.py     # the whole pipeline end to end with RED-CNN
```

## How to run

```bash
# 1. data (see LAPTOP_RUN.md for the download)
bash scripts/rearrange_mayo.sh ~/datasets/mayo_raw
python scripts/prep_mayo.py --src ~/datasets/mayo_raw --dst data/mayo

# 2. quick "does it run at all" test (1 epoch, 5 steps, in runs/<model>_smoke)
python scripts/train.py --model redcnn --smoke

# 3. everything: train (continues after a crash), predict, evaluate
bash scripts/run_all.sh redcnn
```

Or step by step:

```bash
python scripts/train.py    --model redcnn            # add --resume to continue
python scripts/predict.py  --model redcnn            # --sets mayo_L506 lits, --checkpoint best.pth
python scripts/evaluate.py --model redcnn
python scripts/evaluate.py --baseline                # the noisy input itself, for comparison
```

Outputs:

| File | What |
|---|---|
| `runs/<model>/latest.pth` | saved every epoch, used by `--resume` |
| `runs/<model>/best.pth` | the final model (lowest training loss) |
| `runs/<model>/train_log.csv` | epoch, train_loss, val_l1 (HU), lr, seconds |
| `runs/<model>/predictions/<set>/...` | denoised slices (float32 HU) + `timing.json` |
| `runs/<model>/results_<set>.json` | mean and std of every metric |
| `results/summary.csv` | one row per model and test set: **the comparison table** |

## How to add a model

1. `cp ldct/models/template.py ldct/models/<name>.py` and fill in `build()`.
2. Optionally create `configs/models/<name>.yaml` with only the settings that differ
   (e.g. `model_args`, `train.lr`, `infer.patch`).
3. `python scripts/train.py --model <name> --smoke`, then `bash scripts/run_all.sh <name>`.

**The contract:** images go in and come out as `(B, 1, H, W)` in `[0, 1]`.
Training uses 64x64 patches; testing uses whole 512x512 slices
(or tiles, if `infer.patch` is set).

| In your model file | Required? | Used for | Default if missing |
|---|---|---|---|
| `build(**model_args) -> nn.Module` | **yes** | creating the model; `model_args` come from its yaml | – |
| `compute_loss(pred, target, inp)` | no | your own training loss | L1 |
| `training_step(inp, target)` | no | replaces the whole training step (diffusion models) | `model(inp)` + loss |
| `denoise(x)` | no | your own inference for one slice (e.g. diffusion sampling) | `model(x)`, or tiles if `infer.patch` is set |
| `model.epoch` | – | set by `train.py` before every epoch, for losses that change during training | – |

Using an official repo: clone it into `third_party/<Repo>` with a small
`scripts/get_<name>.sh` (pinned to one commit), add that folder to `sys.path` in
your model file, import their network and wrap it. `ldct/models/ct_mamba.py` is
a complete example.
