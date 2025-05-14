python lerobot/scripts/train.py \
  --dataset.repo_id=arclabmit/koch_act_binbox_dataset \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=outputs/train/koch_act_binbox \
  --job_name=koch_act_binbox \
  --policy.device=cuda \
  --wandb.enable=true
