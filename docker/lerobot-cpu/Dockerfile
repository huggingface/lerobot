# Builds CPU-only Docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
FROM python:3.10-slim AS compile-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup virtual environment for Docker
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}

# Make sure we use the virtualenv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install lerobot & specific CPU torch wheel to save on space
COPY . /opt/lerobot
WORKDIR /opt/lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir ".[test, aloha, xarm, pusht]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2
FROM python:3.10-slim AS build-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libegl1-mesa-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv
COPY --from=compile-image /opt/lerobot /lerobot
WORKDIR /lerobot

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/bash"]
