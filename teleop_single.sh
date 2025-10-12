lerobot-teleoperate \
    --robot.type=koch_follower \
    --robot.port=$FOLLOWER_LEFT_PORT \
    --robot.id=follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30} }" \
    --teleop.type=koch_leader \
    --teleop.port=$LEADER_LEFT_PORT \
    --teleop.id=leader \
    --display_data=true
