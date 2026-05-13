#!/bin/bash
#SBATCH --job-name=smolvla2-hirobot
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8

# SmolVLA2 training on an annotated dataset.
#
# The high_level_subtask recipe (recipes/smolvla2_hirobot.yaml) was
# fixed in PR3 to supervise the LM head with the *current* active
# subtask span at every frame, not the next-span target which is
# empty on stable phases. With the old recipe the head learned to
# emit ``\n`` on every chunk boundary; the new one supervises a
# real, scene-grounded string at every frame.
#
# Two regularisers are still on:
#
#   * --dataset.image_transforms.enable=true: torchvision-v2
#     ColorJitter + SharpnessJitter + RandomAffine per frame; default
#     envelope (brightness ±20% etc).
#   * --policy.{plan,memory,subtask}_dropout_prob: randomly drop the
#     context messages carrying the named recipe binding so the model
#     handles missing/stale context. Mirrors Pi0.7 §V.E.

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

DATASET="${DATASET:-pepijn223/super_poulain_full_tool3}"
POLICY_REPO_ID="${POLICY_REPO_ID:-pepijn223/smolvla2_hirobot_super_poulain_tool6}"
JOB_NAME="${JOB_NAME:-smolvla2-hirobot-super-poulain-tool6}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-15000}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/fsx/pepijn/outputs/train/smolvla2_hirobot_super_poulain_tool3_${STEPS}_${RUN_ID}}"

echo "Training smolvla2 on $DATASET"
echo "  GPUs:         $NUM_PROCESSES"
echo "  batch:        $BATCH_SIZE / GPU (global=$((NUM_PROCESSES * BATCH_SIZE)))"
echo "  steps:        $STEPS"
echo "  output:       $OUTPUT_DIR"
echo "  augmentation: image_transforms ON, prompt dropout {plan:0.30 memory:0.30 subtask:0.20}"

accelerate launch --multi_gpu --num_processes="$NUM_PROCESSES" \
    -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla2 \
    --policy.recipe_path=recipes/smolvla2_hirobot.yaml \
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
