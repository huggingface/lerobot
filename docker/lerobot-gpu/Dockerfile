FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    # swig \
    # ffmpeg \
    # libgl1-mesa-glx libosmesa6 libosmesa6-dev libglew-dev mesa-utils libegl-dev \
    # libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    # libegl1-mesa-dev \
    python3-dev python3-pip
    # && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ENV VENV=/opt/venv
# RUN python3 -m venv ${VENV}
# ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

COPY . /lerobot
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[test, aloha, xarm, pusht]"

ENV PYOPENGL_PLATFORM="egl"
ENV MUJOCO_GL="egl"
