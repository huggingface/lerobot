#!/bin/bash

# Example evaluation script for LeRobot policies
unset LEROBOT_HOME
unset HF_LEROBOT_HOME
# === CONFIGURATION ===
POLICY_PATH="ganatrask/lerobot-pi0-libero-object"  # or outputs/train/.../pretrained_model
TASK=libero_object
ENV_TYPE="libero"
BATCH_SIZE=1
N_EPISODES=1
USE_AMP=false
DEVICE=cuda

# === RUN EVALUATION ===
python src/lerobot/scripts/eval.py \
    --policy.path="$POLICY_PATH" \
    --env.type="$ENV_TYPE" \
    --eval.batch_size="$BATCH_SIZE" \
    --eval.n_episodes="$N_EPISODES" \
    --env.multitask_eval=False \
    --env.task=$TASK \
