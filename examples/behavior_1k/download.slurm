#!/bin/bash
#SBATCH -J b1k-download
#SBATCH -p hopper-cpu
#SBATCH --qos=high
#SBATCH -c 64                    # CPUs per conversion (tune as needed)
#SBATCH -t 20:00:00              # Time per conversion
#SBATCH -D /admin/home/francesco_capuano/lerobot
#SBATCH -o /admin/home/francesco_capuano/lerobot/examples/behavior_1k/logs/%x-%A.out
#SBATCH -e /admin/home/francesco_capuano/lerobot/examples/behavior_1k/logs/%x-%A.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# activate your env if needed
source "$HOME/.bashrc" 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi
conda activate lerobot

python examples/behavior_1k/download_data.py \
  --repo-id "behavior-1k/2025-challenge-demos" \
  --local-dir "/fsx/francesco_capuano/behavior1k-2025-v21" \
  --max-workers 64
