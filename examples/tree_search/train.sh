#!/usr/bin/env sh
set -eu

python examples/tree_search/train_reward_model.py \
  --dataset_repo_id="HuggingFaceVLA/libero" \
  --suite="libero_object" \
  --task_orders="all" \
  --episodes_per_task=5 \
  --frame_stride=10 \
  --max_frames_per_episode=32 \
  --scene_camera_key="observation.images.image" \
  --wrist_camera_key="observation.images.image2" \
  --state_key="observation.state" \
  --encoder_type="siglip2" \
  --use_proprioception \
  --proprioception_dim=8 \
  --proprioception_hidden_dim=64 \
  --batch_size=16 \
  --epochs=5 \
  --lr=0.0001 \
  --weight_decay=0.0001 \
  --ranking_weight=0.2 \
  --ranking_margin=0.05 \
  --val_fraction=0.2 \
  --log_every_batches=5 \
  --seed=0 \
  --device="cuda" \
  --output_dir="outputs/tree_search/reward_model_libero_object"
