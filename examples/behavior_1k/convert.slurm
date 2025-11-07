#!/bin/bash
#SBATCH -J b1k-convert
#SBATCH -p hopper-cpu            # pick your partition
#SBATCH --qos=high
#SBATCH --array=0-49%8           # 50 tasks, max 8 running concurrently (conversion is I/O bound)
#SBATCH -c 1                     # CPUs per conversion (tune as needed)
#SBATCH -t 2:00:00               # Time per conversion
#SBATCH --mem=3G                 # ~1.75GB for task 0, ~doubled for safety
#SBATCH -o logs/%x-%A_%a.out     # per-task logs
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # avoid BLAS oversubscription

DATA_PATH="/fsx/francesco_capuano/.cache/behavior-1k/2025-challenge-demos"
BASE_OUT="/fsx/francesco_capuano/behavior1k"
mkdir -p "$BASE_OUT" logs

i="${SLURM_ARRAY_TASK_ID}"
OUT_DIR="$(printf "%s/behavior1k-task%04d" "$BASE_OUT" "$i")"

# activate your env if needed
source ~/.bashrc
conda activate lerobot

python examples/behavior_1k/convert_to_lerobot_v3.py \
  --data-path "$DATA_PATH" \
  --new-repo "$OUT_DIR" \
  --task-id "$i" \
  --force-conversion
