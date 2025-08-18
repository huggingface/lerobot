python -m lerobot.teleoperate \
  --robot.type=so101_follower  --robot.port=/dev/ttyACM1  --robot.id=follower0 \
  --robot.cameras='{ front: {type: opencv, index_or_path: "/dev/video4", width: 1280, height: 720, fps: 30} }' \
  --teleop.type=so101_leader   --teleop.port=/dev/ttyACM0 --teleop.id=leader0 \
  --display_data=false \
  --fps=200