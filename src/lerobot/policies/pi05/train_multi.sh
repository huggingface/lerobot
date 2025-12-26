#!/bin/bash
set -euxo pipefail

# Source YOUR Miniforge conda (mounted from FSX)
source /fsx/jade_choghari/miniforge3/etc/profile.d/conda.sh

conda activate lerobot
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes=8 \
    $(which lerobot-train) \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/data/libero \
    --output_dir=/fsx/jade_choghari/outputs/libero_training_fast_5 \
    --job_name=libero_training_fast \
    --policy.repo_id=jade_choghari/pi05-fast-libero-8 \
    --policy.path=/fsx/jade_choghari/models/libero-pi-fast \
    --policy.dtype=bfloat16 \
    --steps=120000  \
    --save_freq=12000 \
    --batch_size=8 \
    --policy.compile_model=false \
    --policy.device=cuda \
    --policy.fast_only=true \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=120000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=pi05-libero-training \
