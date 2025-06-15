#!/bin/bash

set +x

docker build -t lerobot-gpu-x11 .

# Allow connections from Docker containers to your X server
xhost +local:docker

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the lerobot visualization in a Docker container
docker run -it --rm \
    --gpus all \
    --runtime=nvidia \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --device /dev/ttyACM0 \
    --device /dev/ttyACM1 \
    --shm-size=2gb \
    -e DISPLAY=$DISPLAY \
    -e HF_TOKEN=$HF_TOKEN \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v ${PROJECT_ROOT}:/workspace/lerobot \
    -w /workspace/lerobot \
    -p 3001:3001 \
    lerobot-gpu-x11
    #huggingface/lerobot-gpu

# python lerobot/scripts/visualize_dataset.py --repo-id lerobot/pusht --episode-index 0

# Optional: Revoke X11 permissions after execution (uncomment if desired)
# xhost -local:docker

# ACM0 is follower
# ACM1 is leader