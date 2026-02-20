#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT を設定してください（例: /path/to/airoa-moma）}"

DATASET_REPO_ID="${DATASET_REPO_ID:-airoa-moma-local}"
DATASET_EPISODES="${DATASET_EPISODES:-[0]}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/toyota_hsr_pi0_base_index_embedding}"

HEAD_CAMERA_KEY="${HEAD_CAMERA_KEY:-observation.images.head_camera_rgb}"
HAND_CAMERA_KEY="${HAND_CAMERA_KEY:-observation.images.hand_camera_rgb}"

RENAME_MAP=$(printf '{"%s":"observation.images.right_wrist_0_rgb","%s":"observation.images.left_wrist_0_rgb"}' \
  "$HEAD_CAMERA_KEY" "$HAND_CAMERA_KEY")

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/pi0_base \
  --policy.keep_pretrained_feature_spec=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.observation_rename_map="$RENAME_MAP" \
  --policy.state_action_32_adapter.enabled=true \
  --policy.state_action_32_adapter.mode=index_embedding \
  --policy.state_action_32_adapter.target_state_dim=32 \
  --policy.state_action_32_adapter.target_action_dim=32 \
  --policy.state_action_32_adapter.raw_state_dim=8 \
  --policy.state_action_32_adapter.raw_action_dim=11 \
  --policy.state_action_32_adapter.state_index_map='[0,1,2,3,4,6,11,12]' \
  --policy.state_action_32_adapter.action_index_map='[0,1,2,3,4,6,11,12,13,14,15]' \
  --policy.state_action_32_adapter.apply_mean_std_normalization=true \
  --policy.state_action_32_adapter.disable_builtin_normalizer_for_state_action=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episodes="$DATASET_EPISODES" \
  --batch_size=1 \
  --num_workers=2 \
  --steps=1 \
  --log_freq=1 \
  --output_dir="$OUTPUT_DIR"
