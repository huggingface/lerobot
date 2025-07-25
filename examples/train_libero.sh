#!/bin/bash

# config
REPO_ID=physical-intelligence/libero
TASK=libero_10
OUTPUT_DIR=./outputs/train_run/smolvla2_libero

# clean previous run
rm -rf $OUTPUT_DIR

# ==== Training params ====
STEPS=100000
BATCH_SIZE=4
EVAL_FREQ=2000
SAVE_FREQ=10000
NUM_WORKERS=0

# model params
POLICY=smolvla2
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

echo -e "\033[1;33m[WARNING]\033[0m LIBERO is not yet fully supported in this PR!"
# launch
MUJOCO_GL=egl python src/lerobot/scripts/train.py \
  --policy.type=$POLICY \
  --dataset.repo_id=$REPO_ID \
  --env.type=libero \
  --env.task=$TASK \
  --output_dir=$OUTPUT_DIR \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --num_workers=$NUM_WORKERS \
  --policy.max_action_dim=$MAX_ACTION_DIM \
  --policy.max_state_dim=$MAX_STATE_DIM \
  --policy.use_amp=$USE_AMP \
  --policy.optimizer_lr=$OPTIMIZER_LR \
  --policy.peft_method=$PEFT_METHOD \
  --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
  --policy.repo_id=$VLM_REPO_ID \
  --dataset.use_imagenet_stats=$USE_IMAGENET_STATS \
  --dataset.image_transforms.enable=$ENABLE_IMG_TRANSFORM \
  --dataset.max_num_images=$MAX_NUM_IMAGES \
  --dataset.max_image_dim=$MAX_IMAGE_DIM \
  # --policy.exclude_image_keys=wrist_image \
  --policy.use_env_state=false
