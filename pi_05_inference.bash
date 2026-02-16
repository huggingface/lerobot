rm -rf /home/jeremiah/.cache/huggingface/lerobot/jcoholich/eval_test_lerobot_smolVLA

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.port=/dev/ttyACM0 \
  --robot.id=franka \
  --dataset.single_task="Pick up the orange cup" \
  --dataset.repo_id=jcoholich/eval_test_lerobot_smolVLA \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --policy.dtype=bfloat16 \
  --policy.type=pi05
  # --policy.path=lerobot/pi05_base \