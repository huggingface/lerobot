docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t "${USER}_lerobot" $(dirname "$0")
docker run -it --gpus all --name "${USER}_lerobot_container" \
    -v $(dirname "$0"):/lerobot \
    --device=/dev/ttyDXL_leader_right:/dev/ttyDXL_leader_right \
    --device=/dev/ttyDXL_follower_right:/dev/ttyDXL_follower_right \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    --device=/dev/video2:/dev/video2 \
    --device=/dev/video3:/dev/video3 \
    --device=/dev/video4:/dev/video4 \
    --device=/dev/video5:/dev/video5 \
    --device /dev/snd \
    -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
    -v ${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse \
    "${USER}_lerobot"
