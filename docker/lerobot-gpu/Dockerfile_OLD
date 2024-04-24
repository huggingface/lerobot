# Builds GPU Docker image of PyTorch
# Stage 1
FROM continuumio/miniconda3:latest AS venv-image
ENV PYTHON_VERSION=3.10

# Setup virtual environment
RUN conda create -y -n lerobot python=${PYTHON_VERSION}
ENV PATH=/opt/conda/envs/lerobot/bin:$PATH

# Stage 2
# FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04 AS build-image
# FROM nvidia/opengl:1.0-glvnd-devel-ubuntu22.04 AS build-image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS build-image
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libegl1-mesa-dev \
    libglu1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg swig \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY --from=venv-image /opt/conda/envs/lerobot /opt/conda/envs/lerobot
COPY --from=venv-image /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/lerobot/bin:$PATH

COPY . /lerobot
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[test, aloha, xarm, pusht]"

# Make sure we use the virtualenv
RUN echo ". /etc/profile.d/conda.sh" >> ~/.profile
RUN echo "source activate lerobot" >> ~/.profile
CMD ["/lerobot/docker/bin/docker_entrypoint"]
