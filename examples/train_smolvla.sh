#!/bin/bash


cd ~/lerobot_pi

source ~/miniconda3/bin/activate
conda activate lerobot



export WORK=/home/mustafa_shukor
# export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/transformers
# export HF_HOME=$WORK/.cache/huggingface
# export DATA_DIR=$WORK/.cache/huggingface/datasets
# export HF_LEROBOT_HOME=$WORK/.cache/huggingface/lerobot

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# export WANDB_CACHE_DIR=/lustre/fsn1/projects/rech/dyf/ugz83ue/wandb
# export WANDB_MODE=offline


## then later
## wandb sync wandb/offline-run-*



# V3 So100
REPO_ID=danaaubakirova/svla_so100_task1_v3
DATASET_NAME=so100_v3_task_1


POLICY=smolvla
POLICY_NAME=smolvla




OFFLINE_STEPS=200000
BATCH_SIZE=64


TASK_NAME=lerobot_${DATASET_NAME}_${POLICY_NAME}




TRAIN_DIR=$WORK/logs/lerobot/$TASK_NAME
echo $TRAIN_DIR


rm -r $TRAIN_DIR
CUDA_VISIBLE_DEVICES=2 python lerobot/scripts/train.py \
     --policy.type=$POLICY  \
     --dataset.repo_id=$REPO_ID \
     --output_dir=$TRAIN_DIR
