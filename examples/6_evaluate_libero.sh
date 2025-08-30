#!/bin/bash

unset LEROBOT_HOME
unset HF_LEROBOT_HOME
# CONFIGURATION
POLICY_PATH="ganatrask/lerobot-pi0-libero-object"
TASK=libero_object
ENV_TYPE="libero"
BATCH_SIZE=1
N_EPISODES=1
export MUJOCO_GL=egl
# RUN EVALUATION
python src/lerobot/scripts/eval.py \
    --policy.path="$POLICY_PATH" \
    --env.type="$ENV_TYPE" \
    --eval.batch_size="$BATCH_SIZE" \
    --eval.n_episodes="$N_EPISODES" \
    --env.multitask_eval=True \
    --env.task=$TASK \
