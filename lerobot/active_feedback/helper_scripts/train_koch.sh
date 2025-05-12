python lerobot/scripts/train.py \
  --dataset.repo_id=arclabmit/koch_act_binbox_dataset \
  --policy.type=act \
  --output_dir=outputs/train/koch_act_binbox_model \
  --job_name=koch_act_binbox_model \
  --policy.device=cuda \
  --wandb.enable=true