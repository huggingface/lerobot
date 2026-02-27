#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5

# export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
# export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
# export EGL_DEVICE_ID=0

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # 强制离线记录, 但是lerobot默认是online的, 需要设置wandb.mode=offline
export WANDB_API_KEY=7a17221f579b43949e05faf2a9120c5a6b6506e5
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # /mnt/data_ssd/share/datasets/InternData-A1
    # /mnt/data/daiwanqin/datasets/aloha_sim_transfer_cube_scripted
    # /mnt/data/daiwanqin/datasets/sim/basic_tasks
    # /mnt/data/daiwanqin/datasets/sim/articulation_tasks
    # /mnt/data/daiwanqin/datasets/sim/pick_and_place_tasks/franka/  multiple_pick_and_place_part1/basket
accelerate launch --multi_gpu --num_processes=2 \
 ./src/lerobot/scripts/lerobot_train_multi.py \
    --dataset.root=/mnt/data_ssd/share/datasets/libero \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.streaming=true \
    --dataset.requires_padding=true \
    --policy.type=flower \
    --policy.n_obs_steps=1 \
    --policy.horizon=64 \
    --policy.n_action_steps=60 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=16 \
    --num_workers=4 \
    --steps=200000 \
    --save_freq=10000 \
    --output_dir=outputs/train/train-libero-${TIMESTAMP} \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.mode=offline \
    --eval_freq=10000 \
    --eval.n_episodes=50 \
    --eval.batch_size=16 \
    --env.type=libero \
    --env.task=libero_object,libero_spatial,libero_goal,libero_10 \

# 1. repo_id字符数过长可能造成wandb报错，目前注释掉wandb.init的tag项
# 2. 设置wandb.disable_artifact=true，禁用wandb的artifact功能，避免占用wandb的存储空间
# 3. flower目前使用的vlm只支持长宽相同的图像输入