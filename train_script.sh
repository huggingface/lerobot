#!/bin/bash

# 加载 conda 初始化脚本
# source ~/miniforge3/etc/profile.d/conda.sh

# 激活指定的 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

#HYDRA_FULL_ERROR=1

# 生成当前时间的时间戳
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# 使用时间戳命名日志文件 
log_dir="./logs"
mkdir -p "$log_dir" && chmod 755 "$log_dir"
logfile="${log_dir}/${timestamp}_train.log"

task_name=pick_place_0124_rf10_test

# 使用nohup命令运行脚本并将输出重定向到日志文件
nohup bash -c 'CUDA_VISIBLE_DEVICES=7 python lerobot/scripts/train.py \
--policy.path="/data/jiahuan/huggingface/models/pi0" \
--dataset.repo_id="/data/TR2/hugging_face/pick_place_0124_rf10_test" \
--output_dir="/data/huxian/training/lerobot/pi0_koch_test_$(date +"%m%d_%H%M_%S")" \
--batch_size=8 \
--save_freq=10000 \
--steps=500000 \
--num_workers=12 \
--wandb.enable=true \
--wandb.entity="GBuilders" \
--wandb.project="pi0_TR2" \
--wandb.disable_artifact=true \
--job_name="koch_test_$(date +"%m%d_%H%M_%S")"' > "$logfile" 2>&1 &

# 提示日志文件的位置
echo "Training script is running in the background. Check the log file: $logfile"