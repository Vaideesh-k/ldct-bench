#!/usr/bin/env bash
# One-time setup of the ctmamba conda env (run on a compute node).
set -euo pipefail
export PIP_NO_CACHE_DIR=1
if [ ! -x ~/miniforge3/bin/conda ]; then
  curl -fsSL -o /tmp/miniforge_$USER.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge_$USER.sh -b -p ~/miniforge3
  rm /tmp/miniforge_$USER.sh
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q "^ctmamba " || conda create -y -n ctmamba python=3.10
conda activate ctmamba
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install packaging ninja wheel
export CUDA_HOME=/usr/local/cuda-12.4 MAX_JOBS=8   # only used if a wheel has to be compiled
pip install causal-conv1d==1.6.0 --no-build-isolation
pip install mamba-ssm==2.2.6.post3 --no-build-isolation
pip install -r ~/ldct-bench/requirements.txt
pip install transformers einops timm PyWavelets ptwt ipython matplotlib natsort scipy tqdm pillow
conda clean -afy >/dev/null
echo SETUP_DONE
