FROM huggingface/lerobot-gpu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 vulkan-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[mani-skill]"

# Set EGL as the rendering backend for MuJoCo
ENV MUJOCO_GL="egl"
