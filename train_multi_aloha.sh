#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
export EGL_DEVICE_ID=0

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # 强制离线记录, 但是lerobot默认是online的, 需要设置wandb.mode=offline
export WANDB_API_KEY=7a17221f579b43949e05faf2a9120c5a6b6506e5
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# --dataset.streaming=true \
# accelerate launch --config_file multi_gpu.yaml \
python ./src/lerobot/scripts/lerobot_train_multi.py \
    --dataset.root=/mnt/data/share/datasets/flower/aloha_sim_transfer_cube_scripted \
    --dataset.repo_id=vla-cd/aloha_sim_transfer_cube_scripted \
    --dataset.streaming=true \
    --dataset.requires_padding=true \
    --policy.type=flower \
    --policy.n_obs_steps=1 \
    --policy.horizon=64 \
    --policy.n_action_steps=60 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=32 \
    --num_workers=4 \
    --steps=100000 \
    --save_freq=10000 \
    --valid_freq=-1 \
    --output_dir=./outputs/train-aloha-a1_old-${TIMESTAMP} \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.mode=offline \
    --eval_freq=10000 \
    --eval.n_episodes=50 \
    --eval.batch_size=16 \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
