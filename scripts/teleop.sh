python -m lerobot.teleoperate \
  --robot.type=so101_follower  --robot.port=/dev/ttyACM1  --robot.id=follower0 \
  --teleop.type=so101_leader   --teleop.port=/dev/ttyACM0 --teleop.id=leader0 \
  --fps=200