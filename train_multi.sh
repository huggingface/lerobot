accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=fp16 \
  $(which lerobot-train) \
  --batch_size=32 \
  --save_freq=5000 \
  --num_workers=32 \
  --dataset.repo_id=libero_dataset \
  --dataset.root=/fsx/jade_choghari/datasets/libero/ \
  --policy.type=xvla \
  --output_dir=/fsx/jade_choghari/outputs/train/xvla_libero_multi \
  --job_name=xvla_libero \
  --policy.device=cuda \
  --policy.action_mode=franka_joint7 \
  --wandb.enable=true \
  --policy.repo_id=jadechoghari/X-VLA-Libero \
  --steps=10000
