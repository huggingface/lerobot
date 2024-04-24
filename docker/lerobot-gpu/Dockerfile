FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake swig \
    # git nano rsync vim tree curl \
    # wget unzip htop tmux xvfb patchelf ca-certificates bash-completion libjpeg-dev libpng-dev \
    # libssl-dev libcurl4-openssl-dev libopenmpi-dev  zlib1g-dev \
    # qtbase5-dev qtdeclarative5-dev \
    ffmpeg \
    # libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 \
    libgl1-mesa-glx libosmesa6 libosmesa6-dev libglew-dev mesa-utils libegl-dev \
    libegl1-mesa libegl1-mesa-dev \
    python3-dev python3-pip
    # && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /lerobot
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[test, aloha, xarm, pusht]"

ENV PYOPENGL_PLATFORM="egl"
ENV MUJOCO_GL="egl"
