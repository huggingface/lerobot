#!/bin/bash
#SBATCH -J b1k-upload
#SBATCH -p hopper-cpu
#SBATCH --qos=high
#SBATCH -c 1
#SBATCH -t 48:00:00
#SBATCH --mem=4G
#SBATCH --array=0-49%2
#SBATCH -D /admin/home/francesco_capuano/lerobot
#SBATCH -o /admin/home/francesco_capuano/lerobot/examples/behavior_1k/logs/%x-%A_%a.out
#SBATCH -e /admin/home/francesco_capuano/lerobot/examples/behavior_1k/logs/%x-%A_%a.err

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

# The SLURM_ARRAY_TASK_ID will be used as the task-id
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Configuration
ROOT_PATH="/fsx/francesco_capuano/behavior1k-v3"
HF_USER="fracapuano"
# Limit upload workers to reduce network contention (default in HF Hub is 4)
# For I/O-bound uploads, 2-4 workers per task is optimal
NUM_WORKERS=2

echo "Task ${TASK_ID}: uploading with ${NUM_WORKERS} workers from ${ROOT_PATH}"

python examples/behavior_1k/upload_folders.py \
  --task-id ${TASK_ID} \
  --root-path ${ROOT_PATH} \
  --hf-user ${HF_USER} \
  --num-workers ${NUM_WORKERS}
