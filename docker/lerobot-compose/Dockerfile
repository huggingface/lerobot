FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_CACHE=/root/.cache/pip

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    git git-lfs openssh-client \
    nano vim less util-linux tree \
    htop atop nvtop \
    sed gawk grep curl wget zip unzip \
    tcpdump sysstat screen tmux \
    libglib2.0-0 libgl1 libglx-mesa0 \
    speech-dispatcher \
    # python*-dev is required for building certain python dependencies
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    # Required for cv2.imshow(...) and other highgui operations
    libsm6 \
    # Required for building PyAv \
    pkg-config \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
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

# Setup `python`
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
RUN echo 'if [ "$HOME" != "/root" ]; then ln -sf /root/.local/bin/poetry $HOME/.local/bin/poetry; fi' >> /root/.bashrc

# Copy in pyproject
WORKDIR /lerobot
COPY pyproject.toml poetry.lock ./
# Install dependencies, while using docker builkit cache to speed up future builds
ARG POETRY_EXTRAS="none"
RUN --mount=type=cache,target="${PIP_CACHE}" \
    poetry install --extras ${POETRY_EXTRAS}

ENV PYTHONPATH="/lerobot:${PYTHONPATH}"

# Set the entrypoint such that users enter the container with the poetry environment
ENTRYPOINT ["poetry", "run"]
CMD ["bash"]

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"
