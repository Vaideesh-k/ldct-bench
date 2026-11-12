#!/usr/bin/env bash
# Finish the ctmamba env: official prebuilt mamba wheels + the small packages.
set -euo pipefail
export PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=10
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
W=~/wheels
pip install --no-deps $W/causal_conv1d-1.6.0+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
                      $W/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install -r ~/ldct-bench/requirements.txt
pip install transformers einops timm PyWavelets ptwt ipython matplotlib natsort scipy tqdm pillow
pip check || true
echo SETUP_DONE
