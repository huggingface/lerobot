#!/bin/bash
#SBATCH --job-name=pi052-hirobot-10k
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

DATASET="pepijn223/super_poulain_full_tool3"
POLICY_REPO_ID="pepijn223/pi052_hirobot_super_poulain_tool-g2"
JOB_NAME="pi052-hirobot-super-poulain-tool-g2-10k"
NUM_PROCESSES=8
BATCH_SIZE=32
STEPS=10000
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="/fsx/pepijn/outputs/train/pi052_hirobot_super_poulain_tool3_10k_${RUN_ID}"

echo "Training pi052 on $DATASET with ${NUM_PROCESSES} GPUs, batch size ${BATCH_SIZE}/GPU, ${STEPS} steps"
echo "Output directory: $OUTPUT_DIR"

accelerate launch --multi_gpu --num_processes="$NUM_PROCESSES" \
    -m lerobot.scripts.lerobot_train \
    --policy.type=pi052 \
    --policy.recipe_path=recipes/pi052_hirobot.yaml \
    --dataset.repo_id="$DATASET" \
    --dataset.revision=main \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO_ID" \
    --policy.compile_model=false \
    --policy.device=cuda \
    --policy.tokenizer_max_length=512 \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=hirobot \
    --log_freq=100 \
    --save_freq="$STEPS" \
    --num_workers=0 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --dataset.image_transforms.random_order=true \
    --policy.plan_dropout_prob=0.1 \
    --policy.memory_dropout_prob=0.1 \
    --policy.subtask_dropout_prob=0.1 \
