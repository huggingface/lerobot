FROM huggingface/lerobot-gpu:latest

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install dependencies for building SVT-AV1 and FFmpeg
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    cmake \
    git \
    libass-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libqt5gui5 \
    libsdl2-dev \
    libssl-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    libxcb1-dev \
    nasm \
    ninja-build \
    pkg-config \
    python3 \
    texinfo \
    wget \
    yasm \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /src

# Clone and build SVT-AV1
RUN git clone --depth=1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1 && \
    mkdir -p Build && \
    cd Build && \
    cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd ..

# Set up environment variables for using SVT-AV1 and FFmpeg
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

# Clone and build FFmpeg with SVT-AV1 support
RUN git clone --depth=1 https://github.com/FFmpeg/FFmpeg.git ffmpeg && \
    cd ffmpeg && \
    ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-nonfree \
        --enable-libsvtav1 \
        --disable-doc \
        --disable-htmlpages \
        --disable-manpages \
        --disable-podpages \
        --disable-txtpages && \
    make -j$(nproc) && \
    make install && \
    cd ..

# Use build arguments to pass in user/group IDs
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=lerobot

# Create the user and group
RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}

# Optional: Add the user to sudoers if needed
RUN apt-get update && apt-get install -y sudo && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Set working directory and permissions if needed
WORKDIR /lerobot
RUN chown ${USER_ID}:${GROUP_ID} /lerobot

# Switch to the new user
USER ${USERNAME}
