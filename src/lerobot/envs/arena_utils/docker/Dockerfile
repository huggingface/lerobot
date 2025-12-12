FROM jrottenberg/ffmpeg:7.1-ubuntu2404 AS ffmpeg

FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

# Copy ffmpeg binaries from ffmpeg image
COPY --from=ffmpeg /usr/local /usr/local

# Install necessary dependencies for ffmpeg and uv
RUN apt-get update && apt-get install -y curl libgomp1 git wget openssh-client && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone installer to system-wide location
ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ || true


# Create /devel directory writable by user 1234:1234
RUN mkdir -p /devel && chown -R 1234:1234 /devel && chmod -R 755 /devel
WORKDIR /devel

RUN git clone https://github.com/isaac-sim/IsaacLab.git
WORKDIR /devel/IsaacLab
RUN git checkout v2.3.0
ENV TERM=xterm
RUN ln -s /isaac-sim _isaac_sim && ./isaaclab.sh -i
WORKDIR /devel

# Set uv to use Isaac Sim Python interpreter
ENV UV_PYTHON=/isaac-sim/kit/python/bin/python3

RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:huggingface/collab-lerobot.git
WORKDIR /devel/collab-lerobot
RUN git checkout dev-arena
RUN uv pip install -e .
RUN uv pip install numpy==1.26.0 lxml==4.9.4 packaging==23.2
WORKDIR /devel


RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:isaac-sim/IsaacLab-Arena.git
WORKDIR /devel/IsaacLab-Arena
RUN uv pip install -e .
WORKDIR /devel


# Make Isaac Sim Python the default python/python3
RUN ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python && \
    ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python3 && \
    ln -sf /isaac-sim/kit/python/bin/pip3 /usr/local/bin/pip && \
    ln -sf /isaac-sim/kit/python/bin/pip3 /usr/local/bin/pip3

# Add Isaac Sim Python to PATH
ENV PATH="/isaac-sim/kit/python/bin:${PATH}"
