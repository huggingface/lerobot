#!/bin/bash

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
export CUDA_VISIBLE_DEVICES=3

# CONFIGURATION
POLICY_PATH="/raid/jade/logs/lerobot/lerobot_2_HuggingFaceVLA_libero_smolvla_lr1e-4bs32steps100000/checkpoints/100000/pretrained_model"
POLICY_PATH="AustineJohnBreaker/smolvla_stratch_libero_spatial"
TASK=libero_spatial
ENV_TYPE="libero"
BATCH_SIZE=10
N_EPISODES=10
USE_AMP=false
N_ACTION_STEPS=1
SELF_ATTN_EVERY_N_LAYERS=2
VLM_NAME=HuggingFaceTB/SmolVLM-500M-Instruct
PAD_LANG_TO=longest
LOAD_VLM_WEIGHTS=true
NUM_VLM_LAYERS=16
CHUNK_SIZE=50
N_OBS_STEPS=1
NUM_EXPERT_LAYERS=0
EXPERT_WIDTH_MULTIPLIER=0.5


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
export MUJOCO_GL=egl
ADD_IMAGE_TOKENS=true
unset HF_HUB_OFFLINE
# RUN EVALUATION
python src/lerobot/scripts/eval.py \
    --policy.path="$POLICY_PATH" \
    --env.type="$ENV_TYPE" \
    --eval.batch_size="$BATCH_SIZE" \
    --eval.n_episodes="$N_EPISODES" \
    --env.multitask_eval=False \
    --env.task=$TASK \
    --policy.use_amp=$USE_AMP \
    --policy.n_action_steps=$N_ACTION_STEPS \
    # --policy.add_image_special_tokens=$ADD_IMAGE_TOKENS \
    --policy.attention_mode=$ATTN_MODE \
    --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
    --policy.vlm_model_name=$VLM_NAME \
    --policy.pad_language_to=$PAD_LANG_TO \
    --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
    --policy.num_vlm_layers=$NUM_VLM_LAYERS \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_obs_steps=$N_OBS_STEPS \
    --policy.num_expert_layers=$NUM_EXPERT_LAYERS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH_MULTIPLIER \
