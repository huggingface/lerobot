python src/lerobot/async_inference/robot_openpi_client.py \
    --server_address=localhost:8000 \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
    --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
    --robot.id=bimanual_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30} }" \
    --task="Fold the t-shirt and put it in the bin" \
    --actions_per_chunk=60 \
    --speed_multiplier=1.0
