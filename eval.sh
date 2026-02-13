#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false


export MUJOCO_GL=egl # 强制 MuJoCo 使用 EGL 渲染（关键）
export PYOPENGL_PLATFORM=egl # 禁用 GLFW 图形窗口（避免初始化错误）
export EGL_DEVICE_ID=0


# lerobot-eval \
#   --policy.path="/vla-cd/tmp/daiwanqin/outputs/train-aloha-dataset-20260209_060313/checkpoints/030000/pretrained_model" \
#   --env.type=aloha \
#   --env.task=AlohaTransferCube-v0 \
#   --eval.batch_size=32 \
#   --eval.n_episodes=50

lerobot-eval \
  --policy.path="/vla-cd/tmp/daiwanqin/outputs/train-libero-dataset-20260209_093543/checkpoints/010000/pretrained_model" \
  --env.type=libero \
  --env.task=libero_goal \
  --eval.batch_size=10 \
  --eval.n_episodes=50
