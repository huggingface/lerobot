#!/usr/bin/env bash
set -euo pipefail

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

PYTHON_BIN="${PYTHON_BIN:-python}"

SRC_DATASET_ROOT="${SRC_DATASET_ROOT:-/home/tell/Devenv/ICLR_Competition/airoa-moma}"
PREPARED_DATASET_ROOT="${PREPARED_DATASET_ROOT:-/home/tell/Devenv/ICLR_Competition/airoa-moma-pi05}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-$PREPARED_DATASET_ROOT}"
DATASET_REPO_ID="${DATASET_REPO_ID:-airoa-moma-hsr-pi05-local}"
DATASET_EPISODES="${DATASET_EPISODES:-[0]}"

ACTION_KEY="${ACTION_KEY:-action.relative}"
RUN_INSPECT="${RUN_INSPECT:-true}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
SYMLINK_VIDEOS="${SYMLINK_VIDEOS:-true}"

POLICY_PATH="${POLICY_PATH:-lerobot/pi05_base}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
NORMALIZATION_MAPPING="${NORMALIZATION_MAPPING:-{\"ACTION\":\"QUANTILES\",\"STATE\":\"QUANTILES\",\"VISUAL\":\"IDENTITY\"}}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STEPS="${STEPS:-1}"
LOG_FREQ="${LOG_FREQ:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/toyota_hsr_pi05}"

if [[ ! -f "$SRC_DATASET_ROOT/meta/info.json" ]]; then
  echo "ERROR: SRC_DATASET_ROOT が LeRobot 形式ではありません: $SRC_DATASET_ROOT" >&2
  exit 1
fi

if is_true "$RUN_INSPECT"; then
  "$PYTHON_BIN" scripts/inspect_airoa_moma_features.py \
    --dataset_root "$SRC_DATASET_ROOT" \
    --dataset_repo_id "${INSPECT_DATASET_REPO_ID:-airoa-moma-local}"
fi

if ! is_true "$SKIP_PREPARE"; then
  prepare_args=(
    --src_root "$SRC_DATASET_ROOT"
    --dst_root "$PREPARED_DATASET_ROOT"
    --action_key "$ACTION_KEY"
  )
  if is_true "$FORCE_PREPARE"; then
    prepare_args+=(--force)
  fi
  if is_true "$SYMLINK_VIDEOS"; then
    prepare_args+=(--symlink_videos)
  fi
  "$PYTHON_BIN" examples/toyota_hsr/prepare_airoa_moma_for_pi05.py "${prepare_args[@]}"
fi

if [[ ! -f "$TRAIN_DATASET_ROOT/meta/info.json" ]]; then
  echo "ERROR: TRAIN_DATASET_ROOT に meta/info.json が見つかりません: $TRAIN_DATASET_ROOT" >&2
  exit 1
fi

echo "=== PI0.5 training configuration ==="
echo "POLICY_PATH=$POLICY_PATH"
echo "TRAIN_DATASET_ROOT=$TRAIN_DATASET_ROOT"
echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "DATASET_EPISODES=$DATASET_EPISODES"
echo "ACTION_KEY(for prepare)=$ACTION_KEY"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STEPS=$STEPS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS"

"$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$POLICY_PATH" \
  --policy.keep_pretrained_feature_spec=false \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device="$POLICY_DEVICE" \
  --policy.dtype="$DTYPE" \
  --policy.compile_model="$COMPILE_MODEL" \
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
  --policy.normalization_mapping="$NORMALIZATION_MAPPING" \
  --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$TRAIN_DATASET_ROOT" \
  --dataset.episodes="$DATASET_EPISODES" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUTPUT_DIR"
