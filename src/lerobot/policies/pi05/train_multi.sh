#!/bin/bash
set -euxo pipefail

# Source YOUR Miniforge conda (mounted from FSX)
source /fsx/jade_choghari/miniforge3/etc/profile.d/conda.sh

conda activate lerobot
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes=8 \
    $(which lerobot-train) \
    --dataset.repo_id=local \
    --dataset.root=/fsx/jade_choghari/data/libero \
    --output_dir=/fsx/jade_choghari/outputs/libero_training_fast_mean_1 \
    --job_name=libero_training_fast \
    --policy.repo_id=jade_choghari/pi05-fast-libero \
    --policy.path=/fsx/jade_choghari/models/pi05-base \
    --policy.dtype=bfloat16 \
    --steps=100000 \
    --save_freq=20000 \
    --batch_size=4 \
    --policy.device=cuda \
    --policy.fast_only=true \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
    --rename_map='{
        "observation.images.image1": "observation.images.base_0_rgb",
        "observation.images.image2": "observation.images.left_wrist_0_rgb",
        }' \
    --policy.empty_cameras=1 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=pi05-libero-training \
