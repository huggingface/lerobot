#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,4

# export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
# export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
# export EGL_DEVICE_ID=0

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # 强制离线记录, 但是lerobot默认是online的, 需要设置wandb.mode=offline
export WANDB_API_KEY=7a17221f579b43949e05faf2a9120c5a6b6506e5
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

accelerate launch --config_file multi_gpu.yaml \
 ./src/lerobot/scripts/lerobot_train_multi.py \
    --dataset.root='["/mnt/data/hanmingyan/code/hmy_data_tools/interna1_merge_all/interna1_franka_processed_diff_merge", "/mnt/data/hanmingyan/code/hmy_data_tools/interna1_merge_all/interna1_franka_processed_same_merge"]' \
    --dataset.repo_id='["interna1_merge_all/interna1_franka_processed_diff_merge", "interna1_merge_all/interna1_franka_processed_same_merge"]' \
    --dataset.streaming=true \
    --dataset.requires_padding=true \
    --policy.type=flower \
    --policy.n_obs_steps=1 \
    --policy.horizon=64 \
    --policy.n_action_steps=60 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.vlm_path='/mnt/data/share/models/Florence-2-large' \
    --policy.freeze_embeddings_only=true \
    --policy.load_pretrained=false \
    --policy.pretrained_model_path='/mnt/data_ssd/share/models/flower_vla_pret/360000_model_weights.pt' \
    --policy.resize_h=224 \
    --policy.resize_w=224 \
    --batch_size=64 \
    --num_workers=4 \
    --steps=1600000 \
    --save_freq=20000 \
    --output_dir=./outputs/train-a1-tmp-${TIMESTAMP} \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.mode=offline \