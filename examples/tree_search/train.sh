#!/usr/bin/env sh
set -eu

PYTHONUNBUFFERED=1 uv run python -u examples/tree_search/train_reward_model.py \
  --dataset_repo_id="HuggingFaceVLA/libero,sghosts/noisygain-libero-task0123456789-3pairs_v1" \
  --suite="libero_object" \
  --task_orders="all" \
  --episodes_per_task=5 \
  --frame_stride=10 \
  --max_frames_per_episode=32 \
  --scene_temporal_window=3 \
  --scene_temporal_stride=10 \
  --scene_camera_key="observation.images.image" \
  --wrist_camera_key="observation.images.image2" \
  --state_key="observation.state" \
  --encoder_type="siglip" \
  --use_proprioception \
  --proprioception_dim=8 \
  --proprioception_hidden_dim=64 \
  --batch_size=16 \
  --epochs=5 \
  --lr=0.0001 \
  --weight_decay=0.0001 \
  --ranking_weight=0.2 \
  --ranking_margin=0.05 \
  --no-use_wrong_text_negatives \
  --wrong_text_negatives_per_sample=1 \
  --wrong_text_negative_label=0.0 \
  --bad_sequence_max_reward=0.4 \
  --bad_sequence_decay=4.0 \
  --val_fraction=0.2 \
  --log_every_batches=1 \
  --seed=0 \
  --device="cuda" \
  --output_dir="/content/drive/MyDrive/harezmi-extend-dump/outputs/reward_model_libero_object_temporal_noisy_1"
