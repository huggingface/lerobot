#!/bin/bash
#SBATCH -J b1k-aggregate
#SBATCH -p hopper-cpu
#SBATCH --qos=high
#SBATCH -c 2              # More CPUs won't help much (I/O bound)
#SBATCH -t 20:00:00       # Conservative: 5.2x pattern suggests ~13h, so 12h is tight but reasonable
#SBATCH --mem=8G          # Peak will be ~5-6GB, 8G gives comfortable margin
#SBATCH -o logs/aggregate-50-%j.out
#SBATCH -e logs/aggregate-50-%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

source "$HOME/.bashrc"
conda activate lerobot

cd /admin/home/francesco_capuano/lerobot

python examples/behavior_1k/aggregate_tasks_datasets.py
