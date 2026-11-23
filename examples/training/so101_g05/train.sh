#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
training_python="${PYTHON:-/home/maxime/github/lerobot-g05/.venv/bin/python}"
initial_checkpoint="${INITIAL_CHECKPOINT:-/home/maxime/github/robots/runs/so101_g05_full/initial_checkpoint}"
dataset_root="${DATASET_ROOT:-/home/maxime/github/robots/datasets/so101-fineart-mixed-v1/prepared}"
output_dir="${OUTPUT_DIR:-outputs/train/so101-g05-full-v1}"
"$training_python" -m examples.training.so101_g05.validate \
  --root="$dataset_root" --checkpoint="$initial_checkpoint"
excluded_episodes="$("$training_python" -m examples.training.so101_g05.split --root="$dataset_root")"
exec "$training_python" -m lerobot.scripts.lerobot_train \
  --policy.path="$initial_checkpoint" \
  --dataset.repo_id=maximellerbach/so101-fineart-mixed-v1 --dataset.root="$dataset_root" \
  --dataset.revision=6cd89066f9768dbee8c868af4fa8f108de189089 \
  --dataset.exclude_episodes="$excluded_episodes" \
  --dataset.eval_split=0 --dataset.video_backend=pyav \
  --policy.device=cuda --policy.push_to_hub=false --policy.private=false \
  --batch_size=32 --num_workers=6 --steps=10000 --save_freq=5000 \
  --eval_steps=0 --env_eval_freq=0 --log_freq=25 --wandb.enable=false \
  --output_dir="$output_dir" "$@"
