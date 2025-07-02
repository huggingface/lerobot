#!/bin/bash

RUN_ID=$1

if [ -z "$RUN_ID" ]; then
    echo "Usage: $0 <run_id>"
    exit 1
fi

python train_val.py \
    --policy.type=act \
    --dataset.repo_id=jackvial/merged_datasets_test_2 \
    --output_dir="outputs/train/act_koch_screwdriver_with_validation_${RUN_ID}" \
    --steps=10000 \
    --log_freq=100 \
    --validation.val_freq=200 \
    --validation.enable=true \
    --validation.val_ratio=0.2 \
    --batch_size=8 \
    --wandb.enable=true \
    --wandb.project=self_driving_screwdriver