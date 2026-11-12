#!/bin/bash
# Idea 2 checks on a MIG slice: shapes, parameter counts, memory at batch 32, s/step
# for plain CT-Mamba and the three hybrid variants.   sbatch slurm/hybrid_test.sbatch
#SBATCH --job-name=hybrid_test
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=00:50:00
#SBATCH --output=/dgxa_home/se23uari128/ldct-bench/logs/hybrid_test_%j.out
#SBATCH --error=/dgxa_home/se23uari128/ldct-bench/logs/hybrid_test_%j.err

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ctmamba
cd ~/ldct-bench
echo "host: $(hostname) | date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python tests/check_hybrid.py                      # CPU part
for m in plain low small parallel; do             # GPU part, one process per model
  echo "=== $m"
  python tests/check_hybrid.py --only $m 2>&1 | grep -v Warning | tail -8
done
grep -h "^RESULT" logs/hybrid_test_${SLURM_JOB_ID}.out || true
echo "=== lesion loss / sampling checks"
python tests/check_lesion.py
echo "HYBRID_TEST_OK $(date)"
