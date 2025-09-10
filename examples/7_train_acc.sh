#!/bin/bash
# smolvla training with accelerate

set -euo pipefail

# repo/env
cd ~/lerobot || exit 1
# conda activate lerobot
export LC_ALL=C

rm -f core-*

# storage / caches
RAID=/raid/jade
export TRANSFORMERS_CACHE=$RAID/.cache/huggingface/transformers
export HF_HOME=$RAID/.cache/huggingface
export HF_DATASETS_CACHE=$RAID/.cache/huggingface/datasets
export HF_LEROBOT_HOME=$RAID/.cache/huggingface/lerobot
export WANDB_CACHE_DIR=$RAID/.cache/wandb
export TMPDIR=$RAID/.cache/tmp
mkdir -p $TMPDIR
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

# CONFIG
ENV=libero
TASK=libero_spatial
REPO_ID=physical-intelligence/libero

POLICY=smolvla
VLM=HuggingFaceTB/SmolVLM2-500M-Instruct

# Optim / scheduling
LR=1e-4
DECAY_LR=2.5e-6
DECAY_STEPS=30000
USE_AMP=true   # set to true for mixed precision
TRAIN_EXPERT_ONLY=true
N_ACTION_STEPS=1
SEED=1000

# Training loop
OFFLINE_STEPS=100000
BATCH_SIZE=32
EVAL_FREQ=0
SAVE_FREQ=20000
EVAL_BATCH_SIZE=1
NUM_EPISODES=1

# number of gpus to use
NUM_PROCESSES=2
export CUDA_VISIBLE_DEVICES=1,3
PORT=29522

# naming/output dir
TRAIN_DIR=$RAID/logs/lerobot/lerobot_2_${REPO_ID//\//_}_${POLICY}_lr${LR}bs${BATCH_SIZE}steps${OFFLINE_STEPS}
echo "Training dir: $TRAIN_DIR"

rm -rf "$TRAIN_DIR"

# RUN
python -m accelerate.commands.launch \
  --num_processes $NUM_PROCESSES \
  --num_machines 1 \
  --main_process_port $PORT \
  --mixed_precision=$( [ "$USE_AMP" = true ] && echo "bf16" || echo "no" ) \
  src/lerobot/scripts/train_accelerate.py \
    --policy.type=$POLICY \
    --policy.use_amp=True \
    --policy.vlm_model_name=$VLM \
    --dataset.repo_id=$REPO_ID \
    --dataset.root=$HF_DATASETS_CACHE \
    --env.type=$ENV \
    --env.task=$TASK \
    --output_dir=$TRAIN_DIR \
    --batch_size=$BATCH_SIZE \
    --steps=$OFFLINE_STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval.batch_size=$EVAL_BATCH_SIZE \
    --eval.n_episodes=$NUM_EPISODES \
    --policy.optimizer_lr=$LR \
    --policy.repo_id=None \
    --policy.scheduler_decay_lr=$DECAY_LR \
    --policy.scheduler_decay_steps=$DECAY_STEPS \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.train_expert_only=$TRAIN_EXPERT_ONLY \
    --policy.vlm_model_name=$VLM \
    --seed=$SEED \
    --wandb.enable=false
