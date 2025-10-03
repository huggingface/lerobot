#!/bin/bash
# smolvla training with accelerate

set -euo pipefail


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
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

# CONFIG
ENV=metaworld
TASK=assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3,button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3,coffee-push-v3,dial-turn-v3,disassemble-v3,door-close-v3,door-lock-v3,door-open-v3,door-unlock-v3,drawer-close-v3,drawer-open-v3,faucet-close-v3,faucet-open-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-press-v3,handle-pull-side-v3,handle-pull-v3,lever-pull-v3,peg-insert-side-v3,peg-unplug-side-v3,pick-out-of-hole-v3,pick-place-v3,pick-place-wall-v3,plate-slide-back-side-v3,plate-slide-back-v3,plate-slide-side-v3,plate-slide-v3,push-back-v3,push-v3,push-wall-v3,reach-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3,stick-push-v3,sweep-into-v3,sweep-v3,window-open-v3,window-close-v3
REPO_ID=lerobot/metaworld_mt50
DATASET_NAME=lerobot_metaworld_mt50

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
LOAD_VLM_WEIGHTS=true
# Training loop
OFFLINE_STEPS=100000
BATCH_SIZE=32
EVAL_FREQ=20000
SAVE_FREQ=20000
EVAL_BATCH_SIZE=1
NUM_EPISODES=1
N_OBS_STEPS=1
ATTN_MODE=cross_attn
EXPERT_WIDTH_MULTIPLIER=0.5
# number of gpus to use
NUM_PROCESSES=2
NUM_VLM_LAYERS=0
SELF_ATTN_EVERY_N_LAYERS=0
CHUNK_SIZE=50
export CUDA_VISIBLE_DEVICES=1
PORT=29522
PREFIX_LENGTH=0
LOAD_VLM_WEIGHTS=true
MAX_ACTION_DIM=32
MAX_STATE_DIM=32
# naming/output dir
TRAIN_DIR=$RAID/logs/lerobot/lerobot_new_sep11_v3_${REPO_ID//\//_}_${POLICY}_lr${LR}bs${BATCH_SIZE}steps${OFFLINE_STEPS}
echo "Training dir: $TRAIN_DIR"

rm -rf "$TRAIN_DIR"

lerobot-train \
    --policy.type=$POLICY \
    --policy.use_amp=False \
    --policy.vlm_model_name=$VLM \
    --dataset.repo_id=$REPO_ID \
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
    --policy.n_obs_steps=$N_OBS_STEPS \
    --policy.attention_mode=$ATTN_MODE \
    --policy.prefix_length=$PREFIX_LENGTH \
    --policy.num_vlm_layers=$NUM_VLM_LAYERS \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH_MULTIPLIER \
    --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
    --policy.max_action_dim=$MAX_ACTION_DIM \
    --policy.max_state_dim=$MAX_STATE_DIM \
    --seed=$SEED \
    --wandb.enable=false
