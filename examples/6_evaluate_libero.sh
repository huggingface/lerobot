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
export CUDA_VISIBLE_DEVICES=2

# CONFIGURATION
POLICY_PATH="/raid/jade/logs/lerobot/lerobot_2_HuggingFaceVLA_libero_smolvla_lr1e-4bs32steps100000/checkpoints/100000/pretrained_model"
POLICY_PATH="/raid/jade/models/jade_smolvla2"
TASK=libero_spatial,libero_goal
ENV_TYPE="libero"
BATCH_SIZE=1
N_EPISODES=1
# storage / caches
RAID=/raid/jade
N_ACTION_STEPS=1
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
unset HF_HUB_OFFLINE
# RUN EVALUATION
python src/lerobot/scripts/eval.py \
    --policy.path="$POLICY_PATH" \
    --env.type="$ENV_TYPE" \
    --eval.batch_size="$BATCH_SIZE" \
    --eval.n_episodes="$N_EPISODES" \
    --env.task=$TASK \
    --env.max_parallel_tasks=10 \
# python examples/evaluate_libero.py \
#     --policy_path "$POLICY_PATH" \
#     --task_suite_name "$TASK" \
#     --num_steps_wait 10 \
#     --num_trials_per_task 10 \
#     --video_out_path "data/libero/videos" \
#     --device "cuda" \
#     --seed 7
