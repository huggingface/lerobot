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

# Install pre-built ffmpeg binaries with support for libsvtav1
WORKDIR /ffmpeg
RUN wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
RUN tar -xvf ffmpeg*.tar.xz
RUN mv ffmpeg-master-latest-linux64-gpl/bin/* /usr/local/bin/
# Verify this ffmpeg build has support for libsvtav1, the default lerobot video decoder
RUN ffmpeg -encoders | grep libsvtav1

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
RUN echo 'if [ "$HOME" != "/root" ]; then ln -sf /root/.local/bin/poetry $HOME/.local/bin/poetry; fi' >> /root/.bashrc
RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.in-project true

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
