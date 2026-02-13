#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
# export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
# export EGL_DEVICE_ID=0

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # 需要另外设置wandb.mode=offline
export WANDB_API_KEY=7a17221f579b43949e05faf2a9120c5a6b6506e5
#  --dataset.root=/mnt/data/share/datasets/libero \
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # --dataset.streaming=true \
python ./src/lerobot/scripts/lerobot_train_multi.py \
    --dataset.root=/mnt/data/share/datasets/flower/libero_10_subtask \
    --dataset.repo_id=datasets/libero \
    --dataset.image_transforms.enable=false \
    --dataset.requires_padding=true \
    --policy.type=flower \
    --policy.n_obs_steps=1 \
    --policy.horizon=10 \
    --policy.n_action_steps=10 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=32 \
    --num_workers=4 \
    --steps=100000 \
    --save_freq=10000 \
    --valid_freq=-1 \
    --output_dir=./outputs/train-libero-480k-libero10_dataset-${TIMESTAMP} \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.mode=offline \
    --eval_freq=10000 \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_10 \
    # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}' \

# --dataset.streaming=true \