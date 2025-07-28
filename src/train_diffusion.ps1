python lerobot/scripts/train.py `
  --policy.type=diffusion `
  --policy.repo_id=aiden-li/so101-dp-picktape `
  --dataset.repo_id=aiden-li/so101-picktape `
  --batch_size=128 `
  --steps=120000 `
  --policy.device=cuda `
  --wandb.enable=true `
  --policy.push_to_hub=true `
  --output_dir=outputs/train/so101-dp-picktape `
