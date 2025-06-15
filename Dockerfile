FROM huggingface/lerobot-gpu

RUN apt-get update && apt-get install -y \
    libxkbcommon-x11-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    patchelf \
    libgl1 \
    libglvnd0 \
    libegl1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    mesa-utils \
    mpv \
    --no-install-recommends
RUN pip install transformers

RUN pip install --upgrade mani_skill torch

RUN pip install -e ".[feetech]"

COPY robot_MCP/requirements.txt mcp_requirements.txt

RUN curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash

RUN pip install -e .
RUN pip install keyboard

COPY blue_follower.json /workspace/lerobot/robot_MCP/blue_follower.json
COPY blue_follower.json /root/.cache/huggingface/lerobot/calibration/robots/so101_follower/blue_follower.json

RUN pip install -r mcp_requirements.txt

# Execute the command mcp run robot_MCP/mcp_robot_server.py
#CMD ["mcp", "run", "robot_MCP/mcp_robot_server.py"]

#RUN echo y | python -m mani_skill.utils.download_asset "ReplicaCAD"

# Example command to run the container:
#docker build -t lerobot-gpu-x11 .

# Allow connections from Docker containers to your X server
# xhost +local:docker

# Get the absolute path to the project root directory
# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the lerobot visualization in a Docker container
# docker run -it --rm \
#    --gpus all \
#    --shm-size=2gb \
#    -e DISPLAY=$DISPLAY \
#    -e HF_TOKEN=$HF_TOKEN \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -v $HOME/.Xauthority:/root/.Xauthority:ro \
#    -v ${PROJECT_ROOT}:/workspace/lerobot \
#    -w /workspace/lerobot \
#    lerobot-gpu-x11

# cached location: /root/.cache/huggingface/lerobot/calibration/robots/so101_follower/blue_follower.json
# Leader calibration: -------------------------------------------
# NAME            |    MIN |    POS |    MAX
# shoulder_pan    |    590 |   2169 |   2891
# shoulder_lift   |    938 |    965 |   3257
# elbow_flex      |    936 |   3133 |   3134
# wrist_flex      |    611 |   2758 |   3079
# wrist_roll      |   1031 |   1914 |   3053
# gripper         |   2046 |   2972 |   3005