#!/bin/bash

/workspace/lerobot/lerobot/scripts/train.py \
    --policy.path=lerobot/smolvla_base \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --optimizer.lr=0.001 \
    --dataset.repo_id=a6047425318/green-marker-part2-ep0-debug \
    --dataset.video_backend=pyav \
    --batch_size=4 \
    --steps=2000 \
    --output_dir=outputs/smolvla-overfit-green-marker-part2-ep0-debug \
    --job_name=smolvla-base-overfit-green-marker-part2-ep0-debug \
    --wandb.enable=true \
    --wandb.project=lerobot-training \
    --num_workers=4 \
    --save_checkpoint=false \
    --log_freq=10 \
    --seed=1000 \
    --hub_repo_id=a6047425318/smolvla-overfit-green-marker-part2-ep0-debug
