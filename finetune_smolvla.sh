#!/bin/bash

python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --optimizer.lr=0.0005 \
  --dataset.repo_id=a6047425318/green-marker-part2-ep0-debug \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps=200000 \
  --output_dir=outputs/smolvla-overfit-green-marker-part2-ep0-debug \
  --job_name=smolvla-base-overfit-green-marker-part2-ep0-debug \
  --wandb.enable=true \
  --wandb.project=lerobot-training \
  --num_workers=42 \
  --save_checkpoint=true \
  --save_freq=500 \
  --log_freq=100 \
  --eval_freq=500 \
  --seed=1000 \
  --hub_repo_id="a6047425318/smolvla-overfit-green-marker-part2-ep0-debug"
  # --policy.path="a6047425318/smolvla-overnight-datasets-v1-lr1e-5"



#
#!/bin/bash

python lerobot/scripts/train.py \
  --policy.path=ncavallo/act_so100_lerobot2_block \
  --optimizer.lr=0.00005 \
  --dataset.repo_id=all/datasets \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --steps=20000 \
  --output_dir=outputs/act-all-datasets-manual-override-v1 \
  --job_name=act-all-datasets-manual-override-v1 \
  --wandb.enable=true \
  --wandb.project=lerobot-training \
  --num_workers=32 \
  --save_checkpoint=true \
  --save_freq=500 \
  --log_freq=1 \
  --eval_freq=500 \
  --seed=1000 \
  --hub_repo_id="a6047425318/act-all-datasets-manual-override-v1"
