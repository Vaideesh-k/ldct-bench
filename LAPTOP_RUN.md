# Running CT-Mamba on the NVIDIA laptop

Step by step, with the reason for each step. Run everything **inside Ubuntu**
(WSL terminal), not in Windows PowerShell.

## 0. Before you start

You need the `ctmamba` conda environment with PyTorch + CUDA and the CT-Mamba
dependencies (mamba-ssm, causal-conv1d, timm, einops, PyWavelets, ptwt, torchvision).
Check it:

```bash
conda activate ctmamba
python -c "import torch, mamba_ssm, timm, einops, pywt, ptwt, torchvision; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: a version, `True`, and your GPU name.
*Why:* CT-Mamba only runs on an NVIDIA GPU with mamba-ssm. Finding a missing
package now is faster than finding it after a long data preparation.

## 1. Activate the environment

```bash
conda activate ctmamba
```

*Why:* every command below must use this environment's Python (the one with
PyTorch + CUDA + mamba-ssm), not conda `base`. Do this in every new terminal
and in tmux.

## 2. Get the code

```bash
cd ~
git clone https://github.com/Vaideesh-k/ldct-bench.git
cd ~/ldct-bench
```

*Why:* clone into your Ubuntu home (`~`), **not** under `/mnt/c/...`. Files on
the Windows drive are many times slower to read from Ubuntu, which would slow
down training (it reads thousands of small files).

Later, to get updates: `git pull`.

## 3. Install the small Python packages

```bash
pip install -r requirements.txt
```

*Why:* the pipeline needs numpy, scipy, scikit-image, pydicom, etc.
`requirements.txt` deliberately does **not** list PyTorch, so it will not
replace the CUDA PyTorch already in `ctmamba`.

## 4. Run the tests

```bash
python tests/check_data.py
python tests/check_prep.py
python tests/check_pipeline.py
```

Every line should end in `True` (or `OK`), and `check_pipeline.py` should finish
with no `False`.
*Why:* they use small fake data, so in about a minute they prove that data
loading, DICOM conversion, training, resuming, prediction and evaluation all
work on this laptop **before** you spend hours on real data.
(`check_pipeline.py` trains the small RED-CNN model, not CT-Mamba.)

## 5. Download the Mayo data

From the AAPM Low Dose CT Grand Challenge (Mayo 2016) download, take the
**1 mm, B30 kernel** files:

- `FD_1mm.zip` (full dose = clean)
- `QD_1mm.zip` (quarter dose = noisy)

Put them in `~/datasets/mayo_raw` and unzip:

```bash
mkdir -p ~/datasets/mayo_raw
# move/download FD_1mm.zip and QD_1mm.zip into ~/datasets/mayo_raw, then:
cd ~/datasets/mayo_raw
unzip -q FD_1mm.zip
unzip -q QD_1mm.zip
ls          # expect: full_1mm  quarter_1mm  (plus the two zips)
cd ~/ldct-bench
```

*Why:* 1 mm slices with the B30 kernel are what CT-Mamba was trained and tested
on. Keep the data inside Ubuntu (`~/datasets`), not on `/mnt/c`, for speed.
Check free space first with `df -h ~`.

## 6. Rearrange the folders

```bash
bash scripts/rearrange_mayo.sh ~/datasets/mayo_raw
```

Expected: `Moved 20 folders.` and 10 patients, each with `full_1mm quarter_1mm`.
*Why:* the zips unpack as `quarter_1mm/L067/quarter_1mm/*.IMA`; the next step
wants one folder per patient (`L067/quarter_1mm`, `L067/full_1mm`). Folders are
moved, not copied, so no extra disk space is used. Running it twice is harmless.

## 7. Convert DICOM to .npy

```bash
python scripts/prep_mayo.py --src ~/datasets/mayo_raw --dst data/mayo
```

Expected: **10 lines**, one per patient (L067 ... L506), each saying
`thickness 1.0 / 1.0 mm`.
*Why:* training reads small `.npy` files (int16 HU) much faster than DICOM.
The script also checks that every noisy slice has a clean partner at the same
body position. If a patient is missing or the thickness is not 1.0 mm, you
downloaded the wrong files: stop and tell the team.

## 8. Get the official CT-Mamba code

```bash
bash scripts/get_ct_mamba.sh
```

*Why:* our `ldct/models/ct_mamba.py` only wraps the official code; it does not
copy it. This puts it in `third_party/CT-Mamba` (not in git), pinned to the
exact commit our wrapper was written for.

## 9. Smoke test CT-Mamba

```bash
python scripts/train.py --model ct_mamba --smoke
```

Expected: the parameter count, 5 training steps, one validation, then `Done.`
*Why:* it checks in a minute or two that the GPU, mamba-ssm, memory and the
CT-Mamba loss all work. The first run downloads the ImageNet ResNet-50 (~100 MB)
used by the CT-Mamba loss, so it needs internet. The smoke run is saved in
`runs/ct_mamba_smoke/`, so it never mixes with the real run.

## 10. The full run (inside tmux)

```bash
tmux new -s ctmamba
conda activate ctmamba
cd ~/ldct-bench
mkdir -p runs
bash scripts/run_all.sh ct_mamba 2>&1 | tee -a runs/ct_mamba_console.log
```

Detach with `Ctrl+b`, then `d`. Reattach later with `tmux attach -t ctmamba`.

*Why:* 250 epochs take a long time. tmux keeps the run going if the terminal
window closes. `run_all.sh` trains, then predicts on the test sets with
`best.pth`, then evaluates, and writes `results/summary.csv`.

Keep an eye on it (in another terminal):

```bash
tail -f runs/ct_mamba/train_log.csv     # one line per epoch: loss, val_l1, lr, seconds
nvidia-smi                              # GPU busy and memory use
```

Multiply the `seconds` of one epoch by 250 for the total time. Keep the laptop
plugged in and stop Windows from sleeping (sleep also stops Ubuntu/WSL).

If LiTS is not prepared yet (`data/lits` missing), prediction and evaluation
skip it with a warning and only score Mayo L506. That is expected.

## 11. After a crash or a reboot

Run exactly the same command again:

```bash
tmux new -s ctmamba          # or: tmux attach -t ctmamba
conda activate ctmamba
cd ~/ldct-bench
bash scripts/run_all.sh ct_mamba 2>&1 | tee -a runs/ct_mamba_console.log
```

*Why:* `run_all.sh` always uses `--resume`: training continues after the last
finished epoch saved in `runs/ct_mamba/latest.pth`, with the same optimizer and
learning-rate schedule. At most one epoch is lost.
If it crashed with **CUDA out of memory**, do not change the settings on your
own (that would make the comparison unfair): tell the team first.

## 12. Send the results back

```bash
cp runs/ct_mamba/train_log.csv results/ct_mamba_train_log.csv
git add results/summary.csv results/ct_mamba_train_log.csv
git commit -m "CT-Mamba results on the GPU laptop"
git pull --rebase
git push
```

*Why:* `results/summary.csv` is the comparison table (PSNR, SSIM, RMSE,
seconds per slice, GPU name); the training log shows how the loss developed.
Both are small and allowed in git. (You need collaborator access on GitHub to
push; ask the repo owner.)

Do **not** commit `runs/`, `data/` or `.pth` files: they are big, and CT data is
private (`.gitignore` blocks them). If the team wants the trained model, share
`runs/ct_mamba/best.pth` through a drive link instead.
