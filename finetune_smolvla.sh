#!/bin/bash

python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=danielkorth/whiteboard-marker \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --steps=20000 \
  --output_dir=outputs/whiteboard-and-bike-light-v4 \
  --job_name=whiteboard-and-bike-light-v4 \
  --wandb.enable=true \
  --wandb.project=lerobot-training \
  --num_workers=32 \
  --save_checkpoint=true \
  --save_freq=500 \
  --log_freq=1 \
  --eval_freq=500 \
  --seed=1000 \
  --hub_repo_id="a6047425318/smolvla-whiteboard-and-bike-light-v4"
