# Configure image
ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Install LeRobot
COPY . /lerobot
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[test, aloha, xarm, pusht, dynamixel]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"

# Execute in bash shell rather than python
CMD ["/bin/bash"]
