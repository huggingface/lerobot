





テレオペのコマンド



lerobot-teleoperate --robot.type=so101_follower   --robot.port=/dev/ttyACM0  --robot.id=my_follower_arm --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader_arm



データ収集のコマンド


lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ above: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader_arm \
    --display_data=true \
    --dataset.repo_id=AriRyo/gray-pickplace-v2 \
    --dataset.num_episodes=56 \
    --dataset.single_task="Pick the gray cube and place it on the circle." \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=true














