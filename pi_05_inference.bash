rm -rf /home/jeremiah/.cache/huggingface/lerobot/dummy

export CUDA_VISIBLE_DEVICES=1

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --dataset.push_to_hub=false \
  --dataset.root='/home/jeremiah/.cache/huggingface/lerobot/dummy' \
  --dataset.repo_id=dummy/eval_dummy \
  --dataset.single_task="Pick up the pink block" \
  --dataset.episode_time_s=500 \
  --dataset.num_episodes=10 \
  --policy.dtype=bfloat16 \
  --policy.path=/home/jeremiah/lerobot/outputs/pi05_2_GPU_double_lr/checkpoints/003000/pretrained_model
  # --policy.type=pi05 \