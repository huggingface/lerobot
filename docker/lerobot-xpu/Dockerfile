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
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv --system-site-packages \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

WORKDIR /lerobot
COPY . ./

RUN /opt/venv/bin/pip install --no-cache-dir ".[pusht]"
