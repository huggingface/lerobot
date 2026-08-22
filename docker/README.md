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

### `Dockerfile.rocm` (AMD GPU)

An image for AMD Instinct accelerators, based on a ROCm PyTorch image (`rocm/primus` by default, override with `--build-arg BASE_IMAGE=...`). The base image owns the PyTorch install: LeRobot pins `torch<2.12`, `torchvision<0.27` and `numpy<2.3`, all below what current ROCm images ship, and `[tool.uv.sources]` resolves torch from the CUDA `cu128` index — installing normally would swap in CUDA wheels and leave the GPUs unusable. Those packages are therefore pinned with `uv pip install --override`, the same approach `Dockerfile.benchmark.robomme` uses for its own irreconcilable pins, and the rest of the dependency tree resolves from `pyproject.toml` as usual.

The default base image is compiled for `gfx942`/`gfx950` (MI300X / MI308X / MI325X / MI350X). Other architectures need a base image built for them.

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

# AMD GPU (ROCm)
docker build -f docker/Dockerfile.rocm -t lerobot-rocm .
# ROCm exposes GPUs through `/dev/kfd` and `/dev/dri` rather than `--gpus`, and the container needs the `video` and `render` groups.
docker run -it --rm --ipc=host --shm-size 16gb --device=/dev/kfd --device=/dev/dri --group-add video --group-add render lerobot-rocm
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
