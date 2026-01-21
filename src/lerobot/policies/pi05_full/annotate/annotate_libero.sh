#!/bin/bash

# Example script to run synthetic data generation with Qwen VLM
# This generates user prompts and robot utterances for hierarchical policy training

# Configuration
REPO_ID="lerobot/libero_10"
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
# or: MODEL="Qwen/Qwen2-VL-7B-Instruct"


OUTPUT_DIR="/fsx/jade_choghari/outputs/libero-10-annotate-high"

BATCH_SIZE=16
TEMPERATURE=0.9
SAMPLE_INTERVAL=5.0  # generate dialogue every 1 second (all episodes processed)

# Run subtask annotation
python /admin/home/jade_choghari/lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py \
    --repo-id "$REPO_ID" \
    --video-key observation.images.image \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing \
    --output-repo-id "jadechoghari/libero10-annotate" \
    --batch-size "$BATCH_SIZE" \

# run synthetic data generation (all episodes processed)
# python examples/dataset/annotate_pgen.py \
#     --repo-id "$REPO_ID" \
#     --model "$MODEL" \
#     --output-dir "$OUTPUT_DIR" \
#     --temperature "$TEMPERATURE" \
#     --batch-size "$BATCH_SIZE" \
#     --sample-interval "$SAMPLE_INTERVAL" \
#     --image-key observation.images.base \
#     --num-image-views-per-sample 1

# for faster testing, increase sample interval:
# --sample-interval 5.0  # Samples every 5 seconds (much faster)

# to push to hub after generation:
# add --push-to-hub flag

# efficient batch processing: 4 episodes at once
# python /admin/home/jade_choghari/lerobot/src/lerobot/policies/pi05_full/annotate/high_level_annotate.py \
#     --data-dir "/fsx/jade_choghari/outputs/libero-10-annotate" \
#     --output-dir "$OUTPUT_DIR" \
#     --video-mode \
#     --video-key observation.images.image \
#     --video-batch-size "$BATCH_SIZE" \
#     --sample-interval 5.0
