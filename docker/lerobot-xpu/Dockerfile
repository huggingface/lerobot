FROM intel/intel-extension-for-pytorch:2.7.10-xpu

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa libglu1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1000 -g 100 -m lerobot

USER 1000:100

WORKDIR /home/lerobot
RUN python -m venv ./venv --system-site-packages
COPY --chown=1000:100 --chmod=555 <<EOF ./entrypoint.sh
#!/bin/bash -eu
source /home/lerobot/venv/bin/activate
exec "\$@"
EOF

RUN ./venv/bin/pip install --no-cache-dir PyOpenGL PyOpenGL_accelerate

WORKDIR /lerobot
COPY --chown=1000:100 . ./
RUN /home/lerobot/venv/bin/pip install --no-cache-dir ".[aloha, pusht, smolvla, hilserl]"

ENTRYPOINT ["/home/lerobot/entrypoint.sh"]
