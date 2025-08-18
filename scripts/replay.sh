python -m lerobot.replay \
  --robot.type=so101_follower  --robot.port=/dev/ttyACM1  --robot.id=follower0 \
  --dataset.repo_id=local/so101_test \
  --dataset.episode=0 \
  --play_sounds=false

