FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    git git-lfs openssh-client \
    nano vim less util-linux tree \
    htop atop nvtop \
    sed gawk grep curl wget zip unzip \
    tcpdump sysstat screen tmux \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa \
    speech-dispatcher \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ffmpeg build dependencies. See:
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
# TODO(aliberts): create image to build dependencies from source instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake yasm \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libunistring-dev \
    libmp3lame-dev \
    libtool \
    libvorbis-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    yasm \
    zlib1g-dev \
    nasm \
    libx264-dev \
    libx265-dev libnuma-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libopus-dev \
    libsvtav1-dev libsvtav1enc-dev libsvtav1dec-dev \
    libdav1d-dev

# Install gh cli tool
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Setup `python`
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
RUN echo 'if [ "$HOME" != "/root" ]; then ln -sf /root/.local/bin/poetry $HOME/.local/bin/poetry; fi' >> /root/.bashrc
RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.in-project true

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"
