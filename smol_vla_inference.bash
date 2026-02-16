python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.port=/dev/ttyACM0 \
  --robot.id=franka \
  --robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Grasp a lego block and put it in the bin." \
  --dataset.repo_id=jcoholich/eval_test_lerobot_smolVLA \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --policy.path=lerobot/smolvla_base