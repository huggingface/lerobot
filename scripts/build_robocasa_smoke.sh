#!/bin/bash
# Build a tiny RoboCasa smoke dataset (2 short atomic tasks, all episodes) for
# fast end-to-end training validation before the real run.
#
# Defaults: target/human, OpenStandMixerHead + NavigateKitchen (~1k episodes,
# ~131k frames, ~109 min @ 20 fps), 2 SLURM workers on hopper-cpu.
#
# Override via env: TASKS, REPO_ID, WORK_DIR, WORKERS, CPUS, PARTITION, LOCAL=1.

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

REPO_ID="${REPO_ID:-${HF_USER:?HF_USER is unset}/robocasa_smoke_2atomic_v3}"
WORK_DIR="${WORK_DIR:-/fsx/${USER}/robocasa/datasets/v1.0}"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-/fsx/${USER}/robocasa}"
LOGS_DIR="${LOGS_DIR:-/fsx/${USER}/logs/robocasa}"
TASKS="${TASKS:-OpenStandMixerHead NavigateKitchen}"
WORKERS="${WORKERS:-2}"
CPUS="${CPUS:-8}"
PARTITION="${PARTITION:-hopper-cpu}"
LOCAL="${LOCAL:-0}"

ARGS=(
    examples/port_datasets/slurm_build_robocasa_composite_seen.py
    --repo-id="$REPO_ID"
    --work-dir="$WORK_DIR"
    --robocasa-root="$ROBOCASA_ROOT"
    --split=target --source=human
    --tasks $TASKS
    --workers="$WORKERS"
    --cpus-per-task="$CPUS"
    --partition="$PARTITION"
    --mem-per-cpu=4G
    --time=04:00:00
    --logs-dir="$LOGS_DIR"
    --job-name=port_robocasa_smoke
)
if [[ "$LOCAL" == "1" ]]; then
    ARGS+=(--slurm=0)
fi

echo "Smoke dataset: $REPO_ID"
echo "Tasks: $TASKS"
python "${ARGS[@]}"
