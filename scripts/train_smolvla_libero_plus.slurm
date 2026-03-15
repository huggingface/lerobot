#!/bin/bash
#SBATCH --job-name=smolvla_libero_plus
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/smolvla_libero_plus_%j.out
#SBATCH --error=logs/smolvla_libero_plus_%j.err

set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate lerobot312

cd /admin/home/pepijn/lerobot_wt_robocasa

lerobot-benchmark train \
    --benchmarks libero_plus \
    --policy-path lerobot/smolvla_base \
    --hub-user pepijn223 \
    --num-gpus 4 \
    --steps 30000 \
    --batch-size 32 \
    --eval-freq 0 \
    --wandb \
    --dataset.repo_id=pepijn223/libero_plus_lerobot
