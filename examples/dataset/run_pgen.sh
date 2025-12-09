#!/bin/bash

# Example script to run synthetic data generation with Qwen VLM
# This generates user prompts and robot utterances for hierarchical policy training

# Configuration
REPO_ID="lerobot/svla_so101_pickplace"
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
# Alternative: MODEL="Qwen/Qwen2-VL-7B-Instruct"


OUTPUT_DIR="/fsx/jade_choghari/outputs/pgen_annotations"
BATCH_SIZE=1
TEMPERATURE=0.7
SAMPLE_INTERVAL=1.0  # Generate dialogue every 1 second (all episodes processed)

# Run synthetic data generation (processes ALL episodes)
python examples/dataset/annotate_pgen.py \
    --repo-id "$REPO_ID" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --sample-interval "$SAMPLE_INTERVAL" \
    --num-image-views-per-sample 1

# For faster testing, increase sample interval:
# --sample-interval 5.0  # Samples every 5 seconds (much faster)

# To push to hub after generation:
# Add --push-to-hub flag

