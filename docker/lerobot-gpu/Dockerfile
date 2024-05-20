FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    git git-lfs openssh-client \
    nano vim ffmpeg \
    htop atop nvtop \
    sed gawk grep curl wget \
    tcpdump sysstat screen \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Install LeRobot
RUN git lfs install
RUN git clone https://github.com/huggingface/lerobot.git
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[test, aloha, xarm, pusht]"

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"
