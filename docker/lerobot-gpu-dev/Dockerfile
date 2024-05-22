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

# Setup `python`
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install gh cli tool
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# Install miniconda
RUN mkdir -p /miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh \
    && bash /miniconda3/miniconda.sh -b -u -p /miniconda3 \
    && rm -rf /miniconda3/miniconda.sh

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"
