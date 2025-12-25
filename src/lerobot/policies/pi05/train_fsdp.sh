#!/bin/bash

# FSDP training script for PI05 with aggressive memory optimization
# Use this for large models that OOM with standard DDP

accelerate launch --config_file /admin/home/jade_choghari/lerobot/fsdp_config.yaml \
    $(which lerobot-train) \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/data/libero \
    --output_dir=/fsx/jade_choghari/outputs/libero_training_fsdp \
    --job_name=libero_training_fsdp \
    --policy.repo_id=jade_choghari/pi05-fast-libero-fsdp \
    --policy.path=/fsx/jade_choghari/models/libero-pi-fast \
    --policy.dtype=bfloat16 \
    --steps=100000 \
    --save_freq=10 \
    --batch_size=8 \
    --policy.device=cuda \
    --policy.fast_only=true \
    --policy.scheduler_warmup_steps=2000 \
    --policy.scheduler_decay_steps=60000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=pi05-libero-training-fsdp


