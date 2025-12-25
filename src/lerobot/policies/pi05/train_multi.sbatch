#!/bin/bash
#SBATCH --job-name=pi05-train
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/jade_choghari/logs/%x-%j.out
#SBATCH --error=/fsx/jade_choghari/logs/%x-%j.err

srun \
  --container-image=/fsx/michel_aractingi/docker_images/huggingface+lerobot-gpu+dev.sqsh \
  --container-mounts=/fsx/jade_choghari \
  --container-workdir=$HOME/lerobot \
  bash /admin/home/jade_choghari/lerobot/src/lerobot/policies/pi05/train_multi.sh
