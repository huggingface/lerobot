FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_CACHE=/root/.cache/pip

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic build and dev tools
    build-essential cmake \
    git git-lfs openssh-client \
    nano vim less util-linux tree \
    htop atop nvtop \
    sed gawk grep curl wget zip unzip \
    tcpdump sysstat screen tmux \
    libglib2.0-0 libgl1 libglx-mesa0 \
    # python*-dev is required for building certain python dependencies
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    # Required for cv2.imshow(...) and other highgui operations
    libsm6 \
    # Required for building PyAv \
    pkg-config \
    python3-pip \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    # Symbolic link to python3 \
    python-is-python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pre-built ffmpeg binaries with support for libsvtav1, which is the lerobot default
WORKDIR /ffmpeg
# To update, select a build from https://github.com/BtbN/FFmpeg-Builds/releases
ARG FFMPEG_BUILD=autobuild-2024-10-31-12-59/ffmpeg-N-117676-g87068b9600-linux64-gpl.tar.xz
RUN wget https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD} && \
    tar -xvf ffmpeg*.tar.xz && \
    mv ffmpeg*/bin/* /usr/local/bin/ && \
    # Verify this ffmpeg build succeeded and has support for libsvtav1, the default \
    # lerobot video decoder \
    ffmpeg -encoders | grep libsvtav1 && \
    # Clean up \
    rm -rf /ffmpeg

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
RUN echo 'if [ "$HOME" != "/root" ]; then ln -sf /root/.local/bin/poetry $HOME/.local/bin/poetry; fi' >> /root/.bashrc
RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.in-project true

# Install python dependencies, including `POETRY_EXTRAS` if specified as a --build-arg
WORKDIR /lerobot
COPY pyproject.toml poetry.lock ./
ARG POETRY_EXTRAS="none"
RUN --mount=type=cache,target="${PIP_CACHE}" \
    poetry install --extras ${POETRY_EXTRAS}
ENV PYTHONPATH="/lerobot:${PYTHONPATH}"

# Forcefully reinstall pyav using `--no-binary`, so it has to build itself using the
# systems ffmpeg. This resolves a lockup bug that can happen when trying to open windows
# with OpenCV, with `av` is imported and targets a different version of ffmpeg.
# Tracking issue: https://github.com/opencv/opencv/issues/21952
RUN pip3 install --no-binary pyav --force-reinstall pyav==12.3.0

# Set the entrypoint such that users enter the container with the poetry environment
ENTRYPOINT ["poetry", "run"]
CMD ["bash"]
