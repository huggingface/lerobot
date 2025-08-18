export WANDB_API_KEY=$(cat /ripl/data/projects/lerobot/key.txt)
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
python src/lerobot/scripts/train.py \
  --dataset.repo_id=so101_pick_50 \
  --dataset.root=/ripl/data/projects/lerobot/datasets/so101_pick_50 \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=/ripl/data/projects/lerobot/outputs/train/aug18_pick_50_no_norm \
  --job_name=aug18_pick_50_no_norm \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.project=lerobot \
  --wandb.enable=true
  # --dataset.image_transforms.enable=true \
  # --dataset.image_transforms.max_num_transforms=5 \
  