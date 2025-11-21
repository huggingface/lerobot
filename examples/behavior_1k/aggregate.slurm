#!/bin/bash
#SBATCH -J b1k-aggregate
#SBATCH -p hopper-cpu
#SBATCH --qos=high
#SBATCH -c 2
#SBATCH -t 20:00:00
#SBATCH --mem=4G
#SBATCH -D /admin/home/francesco_capuano/lerobot
#SBATCH -o /admin/home/francesco_capuano/lerobot/examples/behavior_1k/logs/%x-%j.out
#SBATCH -e /admin/home/francesco_capuano/lerobot/examples/behavior_1k/logs/%x-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source "$HOME/.bashrc" 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi
conda activate lerobot

python examples/behavior_1k/aggregate_tasks_datasets.py \
  --task-datasets-dir /fsx/francesco_capuano/behavior1k-v3 \
  --aggregated-root /fsx/francesco_capuano/behavior1k-v3/behavior1k \
  --num-tasks 50 \
  --hf-user fracapuano \
  --push-to-hub
