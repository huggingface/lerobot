#!/bin/bash

echo "=== Local 1-GPU Accelerate Training Test with SmolVLA ==="
echo "Environment: multi"
echo "GPU: 1"
echo "Steps: 50 (quick local test)"
echo ""

# Activate conda environment
source /fsx/dana_aubakirova/miniconda3/etc/profile.d/conda.sh
conda activate multi

# Set CUDA environment for 1 GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Change to working directory
cd /fsx/dana_aubakirova/vla/pr/lerobot

# Set output directory with timestamp
export OUTPUT_DIR="outputs/test_accelerate_1gpu_local_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Test accelerate training with 1 GPU
accelerate launch --config_file accelerate_configs/1gpu_config.yaml -m lerobot.scripts.train \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/svla_so100_sorting \
    --dataset.video_backend=pyav \
    --steps=50 \
    --save_freq=25 \
    --log_freq=5 \
    --batch_size=1 \
    --num_workers=0 \
    --output_dir=$OUTPUT_DIR \
    --wandb.enable=false

echo ""
echo "=== Training completed! ==="
echo "Check outputs in: $OUTPUT_DIR"
