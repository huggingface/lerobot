#!/bin/bash
#SBATCH --job-name=smolvla2-hirobot
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8

# SmolVLA2 training on an annotated dataset, with image augmentation
# and per-component prompt dropout enabled — the two regularisers
# that move the model away from the "text_loss=6e-6 memorised one
# epoch worth of frames" failure mode toward "learns concepts, not
# pixels".
#
# What the regularisers do:
#
#   * --dataset.image_transforms.enable=true: applies torchvision
#     v2 ColorJitter (brightness/contrast/saturation/hue),
#     SharpnessJitter and RandomAffine per frame at training time.
#     Set max_num_transforms to control how many are sampled per
#     frame; defaults to 3 of the 6.
#   * --policy.plan_dropout_prob / memory / subtask: at training,
#     randomly drop the context messages that carry the named
#     binding so the model is forced to handle missing/stale context.
#     Mirrors Pi0.7's prompt-component dropout (§V.E).
#
# Expected effect: text_loss plateaus higher (~0.5-2.0 instead of
# ~1e-5) and the model handles slight prompt/scene drift at
# inference instead of collapsing to memorised fragments.

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

DATASET="${DATASET:-pepijn223/super_poulain_full_tool3}"
POLICY_REPO_ID="${POLICY_REPO_ID:-pepijn223/smolvla2_hirobot_super_poulain_tool4}"
JOB_NAME="${JOB_NAME:-smolvla2-hirobot-super-poulain-tool4}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10000}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/fsx/pepijn/outputs/train/smolvla2_hirobot_${RUN_ID}}"

echo "Training smolvla2 on $DATASET"
echo "  GPUs:         $NUM_PROCESSES"
echo "  batch:        $BATCH_SIZE / GPU (global=$((NUM_PROCESSES * BATCH_SIZE)))"
echo "  steps:        $STEPS"
echo "  output:       $OUTPUT_DIR"
echo "  augmentation: image_transforms ON (wide), prompt dropout {plan:0.20 memory:0.20 subtask:0.30}"

accelerate launch --multi_gpu --num_processes="$NUM_PROCESSES" \
    -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla2 \
    --policy.recipe_path=recipes/smolvla2_hirobot.yaml \
    --dataset.repo_id="$DATASET" \
    --dataset.revision=main \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=4 \
    --dataset.image_transforms.random_order=true \
    --dataset.image_transforms.tfs.brightness.kwargs='{"brightness": [0.5, 1.6]}' \
    --dataset.image_transforms.tfs.contrast.kwargs='{"contrast": [0.6, 1.5]}' \
    --dataset.image_transforms.tfs.saturation.kwargs='{"saturation": [0.3, 1.7]}' \
    --dataset.image_transforms.tfs.hue.kwargs='{"hue": [-0.1, 0.1]}' \
    --dataset.image_transforms.tfs.affine.kwargs='{"degrees": [-15.0, 15.0], "translate": [0.15, 0.15]}' \
    --policy.plan_dropout_prob=0.20 \
    --policy.memory_dropout_prob=0.20 \
    --policy.subtask_dropout_prob=0.30 \
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
    --save_freq=1000 \
    --num_workers=0
