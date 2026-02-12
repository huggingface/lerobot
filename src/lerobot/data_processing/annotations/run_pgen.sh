#!/bin/bash

# Example script to run synthetic data generation with Qwen VLM
# This generates user prompts and robot utterances for hierarchical policy training

# Configuration
REPO_ID="jadechoghari/piper-demo-20260205_103303"
# MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
# or: MODEL="Qwen/Qwen2-VL-7B-Instruct"


OUTPUT_DIR="/fsx/jade_choghari/outputs/collect-data-pgen_new"

BATCH_SIZE=2
TEMPERATURE=0.9
SAMPLE_INTERVAL=5.0  # generate dialogue every 1 second (all episodes processed)

# Run subtask annotation.
# To use closed-vocabulary labels, add a line: --subtask-labels "label1" "label2" ...
# Example (add backslash after "$MODEL" and uncomment the next line):
#   --model "$MODEL" \
#   --subtask-labels "pick_up_yellow_nut_bar" "pick_up_cake" "pick_up_biscuit_pack" "pick_up_soda_can"
python /admin/home/jade_choghari/lerobot/src/lerobot/data_processing/annotations/subtask_annotate.py \
    --repo-id "$REPO_ID" \
    --video-key observation.images.top \
    --output-dir "$OUTPUT_DIR" \
    --output-repo-id "jadechoghari/piper-demo-annotated1" \
    --push-to-hub \
    --model "$MODEL" \
    --subtask-labels "pick_up_yellow_nut_bar" "pick_up_cake" "pick_up_biscuit_pack" "pick_up_soda_can" \
    --batch-size 2

# Run subtask annotation (image-window: frames as images for better accuracy)
# python /admin/home/jade_choghari/lerobot/src/lerobot/data_processing/annotations/subtask_annotate_image.py \
#     --repo-id "$REPO_ID" \
#     --camera-key observation.images.wrist \
#     --output-dir "$OUTPUT_DIR" \
#     --output-repo-id "jadechoghari/piper-demo-annotated1-image" \
#     --push-to-hub \
#     --model "$MODEL" \
#     --window-size 184 \
#     --max-frames-per-window 16 \
#     --subtask-labels "pick_up_yellow_nut_bar" "pick_up_cake" "pick_up_biscuit_pack" "pick_up_soda_can" \
#     --batch-size 2

    
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
