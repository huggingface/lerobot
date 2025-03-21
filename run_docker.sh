docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t "${USER}_lerobot" $(dirname "$0")
docker run -it --gpus all -v $(dirname "$0"):/lerobot --name "${USER}_lerobot_container" "${USER}_lerobot"
