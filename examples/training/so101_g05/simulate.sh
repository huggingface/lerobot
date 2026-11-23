#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec /home/maxime/github/lerobot-g05/.venv/bin/python \
  examples/training/so101_g05/simulate.py \
  --robot.type=so101_sim --robot.episode="${EPISODE:-23}" \
  --policy.path=/home/maxime/github/robots/models/so101-g05-full-v1-pilot-1000 \
  --policy.n_action_steps=32 \
  --device=cuda --strategy.type=base --inference.type=sync \
  --interactive=true --fps=30 --return_to_initial_position=false --play_sounds=false \
  --task="Put the orange cube in the gray box." "$@"
