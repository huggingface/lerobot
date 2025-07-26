#!/bin/bash

# config
OUTPUT_DIR=./outputs/train_run
REPO_IDS=pranavsaroha/so100_onelego2,pranavsaroha/so100_onelego3,pranavsaroha/so100_carrot_2
VLM_REPO_ID=None
STEPS=100000
BATCH_SIZE=8
EVAL_FREQ=5000
NUM_WORKERS=0

# model config
POLICY=smolvla2
USE_AMP=false
OPTIMIZER_LR=1e-4
PEFT_METHOD=lora
LOAD_VLM_WEIGHTS=true
MAX_ACTION_DIM=32
MAX_STATE_DIM=32

# dataset config
USE_IMAGENET_STATS=false
ENABLE_IMG_TRANSFORM=true
MAX_NUM_IMAGES=2
MAX_IMAGE_DIM=1920

# launch
python src/lerobot/scripts/train.py \
  --policy.type=$POLICY \
  --dataset.repo_id=$REPO_IDS \
  --dataset.use_imagenet_stats=$USE_IMAGENET_STATS \
  --dataset.image_transforms.enable=$ENABLE_IMG_TRANSFORM \
  --policy.max_action_dim=$MAX_ACTION_DIM \
  --policy.max_state_dim=$MAX_STATE_DIM \
  --output_dir=$OUTPUT_DIR \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --eval_freq=$EVAL_FREQ \
  --policy.use_amp=$USE_AMP \
  --policy.optimizer_lr=$OPTIMIZER_LR \
  --policy.peft_method=$PEFT_METHOD \
  --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
  --policy.repo_id=$VLM_REPO_ID \
  --dataset.max_num_images=$MAX_NUM_IMAGES \
  --dataset.max_image_dim=$MAX_IMAGE_DIM \
  --num_workers=$NUM_WORKERS
