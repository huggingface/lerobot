# Docker

This directory contains Dockerfiles for running LeRobot in containerized environments. Both images are **built nightly from `main`** and published to Docker Hub with the full environment pre-baked — no dependency setup required.

## Pre-built Images

```bash
# CPU-only image (based on Dockerfile.user)
docker pull huggingface/lerobot-cpu:latest

# GPU image with CUDA support (based on Dockerfile.internal)
docker pull huggingface/lerobot-gpu:latest
```

## Quick Start

The fastest way to start training is to pull the GPU image and run `lerobot-train` directly. This is the same environment used for all of our CI, so it is a well-tested, batteries-included setup.

```bash
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest

# inside the container:
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
```

## Dockerfiles

### `Dockerfile.user` (CPU)

A lightweight image based on `python:3.12-slim`. Includes all Python dependencies and system libraries but does not include CUDA — there is no GPU support. Useful for exploring the codebase, running scripts, or working with robots, but not practical for training.

### `Dockerfile.internal` (GPU)

A CUDA-enabled image based on `nvidia/cuda`. This is the image for training — mostly used for internal interactions with the GPU cluster.

## Usage

### Running a pre-built image

```bash
# CPU
docker run -it --rm huggingface/lerobot-cpu:latest

# GPU
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest
```

### Building locally

From the repo root:

```bash
# CPU
docker build -f docker/Dockerfile.user -t lerobot-user .
docker run -it --rm lerobot-user

# GPU
docker build -f docker/Dockerfile.internal -t lerobot-internal .
docker run -it --rm --gpus all --shm-size 16gb lerobot-internal
```

### Multi-GPU training

To select specific GPUs, set `CUDA_VISIBLE_DEVICES` when launching the container:

```bash
# Use 4 GPUs
docker run -it --rm --gpus all --shm-size 16gb \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  huggingface/lerobot-gpu:latest
```

### USB device access (e.g. robots, cameras)

```bash
docker run -it --device=/dev/ -v /dev/:/dev/ --rm huggingface/lerobot-cpu:latest
```
