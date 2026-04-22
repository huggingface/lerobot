#!/bin/bash
#SBATCH -A gts-agarg35
#SBATCH -N1
#SBATCH --mem-per-gpu=60G
#SBATCH -q embers
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu-a100
#SBATCH -o /storage/project/r-agarg35-0/igeorgiev3/lerobot/slurm_eval_%j.log
#SBATCH -e /storage/project/r-agarg35-0/igeorgiev3/lerobot/slurm_eval_%j.log
#SBATCH -J fastwam_a100

export HF_HOME=/storage/project/r-agarg35-0/shared/huggingface_cache
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH=/storage/project/r-agarg35-0/shared/awm/fastwam_wan22_weights
export DIFFSYNTH_SKIP_DOWNLOAD=false
export MUJOCO_GL=egl

source /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /storage/project/r-agarg35-0/igeorgiev3/lerobot

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== FastWAM eval on A100: 3 episodes ==="
echo N | lerobot-eval \
    --policy.path=/storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_10 \
    --env.observation_height=224 \
    --env.observation_width=224 \
    --eval.batch_size=1 \
    --eval.n_episodes=3 \
    --policy.num_inference_steps=10

echo "=== Eval exit code: $? ==="
