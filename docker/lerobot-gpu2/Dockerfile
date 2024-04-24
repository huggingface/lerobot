FROM registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/cuda:12.2.2-base-ubuntu20.04-libegl1

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential git nano rsync vim tree curl \
wget unzip htop tmux xvfb patchelf ca-certificates bash-completion libjpeg-dev libpng-dev \
ffmpeg cmake swig libssl-dev libcurl4-openssl-dev libopenmpi-dev python3-dev zlib1g-dev \
qtbase5-dev qtdeclarative5-dev libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 \
libgl1-mesa-glx libosmesa6 libosmesa6-dev libglew-dev mesa-utils libegl-dev \
libegl1-mesa libegl1-mesa-dev python3-pip

ENV PYOPENGL_PLATFORM="egl"
ENV MUJOCO_GL="egl"
