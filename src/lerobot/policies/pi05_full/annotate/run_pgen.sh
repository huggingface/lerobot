#!/bin/bash

# Example script to run synthetic data generation with Qwen VLM
# This generates user prompts and robot utterances for hierarchical policy training

# Configuration
REPO_ID="jadechoghari/collect-data"
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
# or: MODEL="Qwen/Qwen2-VL-7B-Instruct"


OUTPUT_DIR="/fsx/jade_choghari/outputs/collect-data-pgen_new"

BATCH_SIZE=32
TEMPERATURE=0.9
SAMPLE_INTERVAL=5.0  # generate dialogue every 1 second (all episodes processed)

# Run subtask annotation
python /admin/home/jade_choghari/lerobot/src/lerobot/policies/pi05_full/annotate/subtask_annotate.py \
    --repo-id "$REPO_ID" \
    --video-key observation.images.base \
    --output-dir "$OUTPUT_DIR" \
    --output-repo-id "jadechoghari/collect-data-with-subtasks"
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
# python examples/dataset/annotate_pgen.py \
#     --repo-id "$REPO_ID" \
#     --model "$MODEL" \
#     --output-dir "$OUTPUT_DIR" \
#     --video-mode \
#     --video-key observation.images.up \
#     --video-batch-size "$BATCH_SIZE" \
#     --sample-interval 1.0
