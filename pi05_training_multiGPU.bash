#!/bin/bash
#SBATCH --job-name=pi05_training
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:2
#SBATCH --cpus-per-gpu=10
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=8G
#SBATCH -x baymax

JOB_NAME=$1
OUTDIR=./outputs/$JOB_NAME

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
LR=$(awk -v n="$NUM_GPUS" 'BEGIN { printf "%.10g", 2.5e-5 * n }')

nvidia-smi

echo "Using $NUM_GPUS GPUs with a learning rate of $LR"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot
accelerate launch \
--multi_gpu \
--num_processes=$NUM_GPUS \
--mixed_precision=bf16 \
$(which lerobot-train) \
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
    --policy.optimizer_lr=$LR \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=24 \
    --log_freq=5
