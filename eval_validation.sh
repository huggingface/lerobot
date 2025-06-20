#!/bin/bash

# Validation evaluation for SmolVLA models
# Usage: ./eval_validation.sh <path_to_checkpoint>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_checkpoint>"
    echo "Example: $0 outputs/whiteboard-and-bike-light-v5-lr1e-5/checkpoints/step_020000/pretrained_model"
    exit 1
fi

CHECKPOINT_PATH=$1

echo "Running validation evaluation on checkpoint: $CHECKPOINT_PATH"

python lerobot/evaluate_smolvla_validation.py \
  --policy.path="$CHECKPOINT_PATH" \
  --dataset.repo_id=all/datasets \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --policy.device=cuda
