python -m lerobot.record \
  --robot.type=so101_follower  --robot.port=/dev/ttyACM1  --robot.id=follower0 \
  --teleop.type=so101_leader   --teleop.port=/dev/ttyACM0 --teleop.id=leader0 \
  --robot.cameras='{ front: {type: opencv, index_or_path: "/dev/video4", width: 640, height: 480, fps: 30} }' \
  --display_data=false \
  --dataset.repo_id=local/so101_pick_10 \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick up the wooden cube and put it in the circle" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false