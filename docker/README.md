# Docker Environment
## How to Use
### (Optional) Set wandb key
Note: `wandb_key.txt` is git-ignored
```bash
cp wandb_key.txt.keep wandb_key.txt
echo '{YOUR_WANDB_KEY}' >> wandb_key.txt
```

### (Optional) Setup `DATASETS_PATH` to mount into the container
```bash
$ export DATASETS_PATH={YOUR_HOST_DATASETS_PATH_TO_MOUNT}
```
The directory is mounted as `/root/datasets` in the container.

### 1. Build docker image
Creating an image named `{YOUR_HOST_NAME}/lerobot:latest`
```bash
./BUILD_DOCKER_IMAGE.sh
```

### 2. Run docker container (takes some time to install additional packages)
Creating a container named `{YOUR_HOST_NAME}_gnfactor`
```bash
./RUN_DOCKER_CONTAINER.sh
```

### 3. Enter the container
```bash
docker exec -it {YOUR_HOST_NAME}_lerobot bash
```

### 4. Move to `/root/lerobot` directory in the container where this repository (lerobot) is mounted
```bash
cd /root/lerobot
```

