#!/bin/sh
xhost +

if [ ! -z $1 ]; then
  export TAG_NAME=$1
else
  export TAG_NAME="lerobot-gpu-dev"
fi

# Setting up dataset path
if [ -n "$DATASETS_PATH" ]; then
  echo "DATASETS_PATH is set to '$DATASETS_PATH'. Mounting this path to the container as '/root/datasets'."
else
  export DATASETS_PATH="$PWD/../datasets"
  echo "DATASETS_PATH is not set. Using default path: $DATASETS_PATH"
fi

# Setting up wandb key
export WANDB_API_KEY=`cat wandb_key.txt`

docker-compose -p lerobot up &

