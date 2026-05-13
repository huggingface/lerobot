#!/bin/bash
#SBATCH --job-name=pi052-hirobot
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8

# π0.5 v2 training — reproduces the π0.5 paper's hierarchical recipe.
#
# Same recipe blend as the SmolVLA2 stack (recipes/pi052_hirobot.yaml),
# just on the PaliGemma 2B + Gemma-300m action-expert backbone the
# paper uses. The text head learns subtask prediction via cross-
# entropy on supervised spans; the action expert learns the flow
# field. Paper §IV.D mixes the two losses with α=10, which we encode
# as flow_loss_weight=10 / text_loss_weight=1.

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

DATASET="${DATASET:-pepijn223/super_poulain_full_tool3}"
POLICY_REPO_ID="${POLICY_REPO_ID:-pepijn223/pi052_hirobot_super_poulain}"
JOB_NAME="${JOB_NAME:-pi052-hirobot-super-poulain}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-15000}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/fsx/pepijn/outputs/train/pi052_hirobot_${STEPS}_${RUN_ID}}"

echo "Training pi052 on $DATASET"
echo "  GPUs:         $NUM_PROCESSES"
echo "  batch:        $BATCH_SIZE / GPU (global=$((NUM_PROCESSES * BATCH_SIZE)))"
echo "  steps:        $STEPS"
echo "  output:       $OUTPUT_DIR"
echo "  loss mix:     flow_loss_weight=10 (paper α), text_loss_weight=1"
echo "  augmentation: image_transforms ON, prompt dropout {plan:0.30 memory:0.30 subtask:0.20}"

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
    --policy.text_loss_weight=1.0 \
    --policy.flow_loss_weight=10.0 \
    --policy.unfreeze_lm_head=true \
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
    --policy.plan_dropout_prob=0.30 \
    --policy.memory_dropout_prob=0.30 \
    --policy.subtask_dropout_prob=0.20
