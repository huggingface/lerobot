python src/lerobot/async_inference/robot_openpi_client.py \
    --server_address=localhost:8000 \
    --robot.type=koch_follower \
    --robot.port=$FOLLOWER_LEFT_PORT \
    --robot.id=follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}}" \
    --actions_per_chunk=25
