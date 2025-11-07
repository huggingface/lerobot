lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=xvla \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=xvla_so101_pickplace \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=jadechoghari/xvla_policy \
  --steps=10000
