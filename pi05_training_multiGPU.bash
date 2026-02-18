#!/bin/bash
#SBATCH --job-name=pi05_training
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:2
#SBATCH -c 15
#SBATCH --qos=long

JOB_NAME=$1
OUTDIR=./outputs/$JOB_NAME

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot
accelerate launch \
--multi_gpu \
--num_processes=2 \
--mixed-precision=bf16 \
src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=eve_blocks \
    --dataset.root='/coc/testnvme/jcoholich3/lerobot_data/eve_blocks' \
    --policy.type=pi05 \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=24 \
    --log_freq=5
