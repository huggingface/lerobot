lerobot-train \
  --dataset.repo_id=libero_dataset \
  --dataset.root=/fsx/jade_choghari/datasets/libero/ \
  --policy.type=xvla \
  --output_dir=/fsx/jade_choghari/outputs/train/xvla_libero \
  --job_name=xvla_libero \
  --policy.device=cuda \
  --policy.action_mode=franka_joint7 \
  --wandb.enable=true \
  --policy.repo_id=jadechoghari/X-VLA-Libero \
  --steps=10000

#   # --policy.pretrained_path=/fsx/jade_choghari/.cache/huggingface/model/xvla-libero \