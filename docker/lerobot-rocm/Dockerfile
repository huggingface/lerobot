# Base image with ROCm, PyTorch, and Ubuntu 22.04
FROM rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.10_pytorch_release_2.4.0

WORKDIR /workspace

# Configure user groups for GPU access
RUN groupadd -f render && \
    usermod -aG render root && \
    usermod -aG video root

# Install system dependencies
RUN apt-get update && apt-get install -y \
        git \
        cmake \
        build-essential \
        python3-dev \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
        ffmpeg \
        speech-dispatcher && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone and install LeRobot
RUN git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    sed -i '/"torch[>=<~ ]*/d' pyproject.toml && \
    sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml && \
    pip install -e ".[aloha,pusht,dynamixel]"

# Install additional Python packages
RUN pip install --no-cache-dir \
        "numpy<2" \
        pytest \
        transformers \
        huggingface_hub
