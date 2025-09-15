#!/bin/bash

# config
REPO_ID=HuggingfaceVLA/libero
TASK=libero_10,libero_spatial
OUTPUT_DIR=./outputs/

# clean previous run
rm -rf $OUTPUT_DIR

# training params
STEPS=100000
BATCH_SIZE=4
EVAL_FREQ=1
SAVE_FREQ=10000
NUM_WORKERS=4

# model params
POLICY=smolvla
USE_AMP=false
OPTIMIZER_LR=1e-4
PEFT_METHOD=lora
LOAD_VLM_WEIGHTS=true
VLM_REPO_ID=None
MAX_ACTION_DIM=32
MAX_STATE_DIM=32

# dataset/image params
USE_IMAGENET_STATS=false
ENABLE_IMG_TRANSFORM=true
MAX_NUM_IMAGES=2
MAX_IMAGE_DIM=1024
unset LEROBOT_HOME
unset HF_LEROBOT_HOME
export MUJOCO_GL=egl
echo -e "\033[1;33m[WARNING]\033[0m LIBERO is not yet fully supported in this PR!"

# launch
python src/lerobot/scripts/train.py \
  --policy.type=$POLICY \
  --dataset.repo_id=$REPO_ID \
  --dataset.root='/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data' \
  --env.type=libero \
  --env.task=$TASK \
  --output_dir=$OUTPUT_DIR \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --num_workers=$NUM_WORKERS \
  --policy.repo_id=$VLM_REPO_ID \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.repo_id=None \
