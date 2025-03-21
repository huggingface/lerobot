FROM huggingface/lerobot-gpu:latest

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
