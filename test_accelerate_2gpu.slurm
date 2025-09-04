#!/bin/bash
#SBATCH --job-name=test_accelerate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/dana_aubakirova/vla/logs/test_accelerate_%j.out
#SBATCH --error=/fsx/dana_aubakirova/vla/logs/test_accelerate_%j.err

# Create logs directory if it doesn't exist
mkdir -p /fsx/dana_aubakirova/vla/pr/lerobot/logs

# Activate conda environment
source /fsx/dana_aubakirova/miniconda3/etc/profile.d/conda.sh
conda activate multi

# 2-GPU Test CUDA environment
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false
export HF_ACCELERATE_DEVICE_MAP=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export SAFETENSORS_FAST_GPU=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_TORCH_DEVICE_MAP_AUTO=false

# Change to working directory
cd /fsx/dana_aubakirova/vla/pr/lerobot

echo "=== Testing Accelerate Multi-GPU Training with SmolVLA ==="
echo "Dataset: lerobot/svla_so100_sorting"
echo "GPUs: 2"
echo "Steps: 100 (for quick test)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Set output directory with job ID
export OUTPUT_DIR="outputs/test_accelerate_2gpu_job_${SLURM_JOB_ID}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Test accelerate training
accelerate launch --config_file accelerate_configs/2gpu_config_safe.yaml -m lerobot.scripts.train \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/svla_so100_sorting \
    --dataset.video_backend=pyav \
    --steps=100 \
    --save_freq=50 \
    --log_freq=5 \
    --batch_size=2 \
    --num_workers=0 \
    --output_dir=$OUTPUT_DIR \
    --wandb.enable=false

echo ""
echo "=== Training completed! ==="
echo "Check logs and outputs in: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
