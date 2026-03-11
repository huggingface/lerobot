#!/bin/bash
# Example script to run subtask annotation with Qwen VLM.
# Generates skill/subtask segments for hierarchical policy training.
#
# Usage:
#   ./scripts/run_subtask_annotate.sh
# Or with env overrides:
#   REPO_ID=user/other-dataset OUTPUT_DIR=/tmp/out ./scripts/run_subtask_annotate.sh

set -e

# --------------- Configuration ---------------
REPO_ID="${REPO_ID:-lerobot-data-collection/round1_4}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
# Or: MODEL="Qwen/Qwen2-VL-7B-Instruct"

OUTPUT_DIR="${OUTPUT_DIR:-/fsx/jade_choghari/outputs/collect-data-pgen_new}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VIDEO_KEY="${VIDEO_KEY:-observation.images.base}"

# Closed-vocabulary subtask labels (optional). Pass as space-separated list.
# Leave empty for open-vocabulary segmentation.
SUBTASK_LABELS=(
  "do_first_horizontal_fold"
  "do_second_horizontal_fold"
  "do_third_fold_left_to_right"
  "do_fourth_fold_right_to_left"
  "rotate_t-shirt_90_degrees"
  "slide_folded_shirt_to_right"
)

# --------------- Run subtask annotation ---------------
# Uses the refactored CLI: lerobot-dataset-subtask-annotate (config-based, snake_case args).
# If not installed, run from repo root: PYTHONPATH=src python -m lerobot.scripts.lerobot_subtask_annotate ...

CMD=(
  lerobot-dataset-subtask-annotate
  --repo_id "$REPO_ID"
  --video_key "$VIDEO_KEY"
  --output_dir "$OUTPUT_DIR"
  --push_to_hub True
  --no_timer_overlay True
  --model "$MODEL"
  --batch_size "$BATCH_SIZE"
)

# Add closed-vocabulary labels if defined
if [ ${#SUBTASK_LABELS[@]} -gt 0 ]; then
  for label in "${SUBTASK_LABELS[@]}"; do
    CMD+=(--subtask_labels "$label")
  done
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# --------------- Optional: image-window annotator (if available) ---------------
# Uncomment and adjust path if you use the image-window variant:
# python /path/to/lerobot/data_processing/annotations/subtask_annotate_image.py \
#   --repo-id "$REPO_ID" \
#   --camera-key observation.images.wrist \
#   --output-dir "$OUTPUT_DIR" \
#   --output-repo-id "jadechoghari/piper-demo-annotated1-image" \
#   --push-to-hub \
#   --model "$MODEL" \
#   --window-size 184 \
#   --max-frames-per-window 16 \
#   --subtask-labels "label1" "label2" \
#   --batch-size 2

# --------------- Optional: synthetic data generation (pgen) ---------------
# BATCH_SIZE=2 TEMPERATURE=0.9 SAMPLE_INTERVAL=5.0
# python examples/dataset/annotate_pgen.py \
#   --repo-id "$REPO_ID" \
#   --model "$MODEL" \
#   --output-dir "$OUTPUT_DIR" \
#   --temperature "$TEMPERATURE" \
#   --batch-size "$BATCH_SIZE" \
#   --sample-interval "$SAMPLE_INTERVAL" \
#   --image-key observation.images.base \
#   --num-image-views-per-sample 1
# Add --push-to-hub to push after generation.
