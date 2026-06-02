#!/bin/bash
#SBATCH --job-name=pi052-hirobot-robocasa-human300
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export LEROBOT_DEBUG_PREDS_EVERY=1000
export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Compile path: pin triton + inductor caches node-local. The shared
# /fsx cache mixes kernels built against different glibc versions and
# trips ``GLIBC_2.34 not found`` on hopper nodes (bench v3 confirmed).
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Non-fatal so an unstaged local hotfix doesn't kill the job. CI / clean
# checkouts still fast-forward as before; dirty trees just keep their
# in-flight changes (the working tree is what runs).
git pull --ff-only || echo "[warn] git pull skipped — keeping working tree."
python -m pip install -q --upgrade -e .
python -m pip install -q --upgrade -e '.[pi]'
python -m pip install -q --upgrade 'liger-kernel'

# FlashAttention-2 is NOT installed. The pi052 dual-expert layer compute
# uses SDPA (the block-bidirectional mask is unsupported by FA2 anyway),
# and the only other consumer would be liger-kernel — which gracefully
# degrades when flash_attn is absent. The previously-installed wheel was
# built against a newer GLIBC than some hopper compute nodes provide
# (job 22162586 on ip-26-0-162-14 hit ``GLIBC_2.32 not found``), so the
# safest configuration is "not installed". To re-enable for the
# downstream HF Gemma ``generate`` path, install a wheel matching the
# node's libc — but verify on every assigned node first.

DATASET="pepijn223/robocasa_pretrain_human300_v4"
DATASET_REVISION="${DATASET_REVISION:-main}"
POLICY_REPO_ID="pepijn223/pi052_robocasa_human300"
JOB_NAME="pi052-hirobot-robocasa-human300"
NUM_PROCESSES=8
# BS=36 — fits ~72 GB / 80 GB, BS=36 × 8 GPUs = 288 effective.
BATCH_SIZE=${BATCH_SIZE:-36}
STEPS=${STEPS:-5000}
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="/fsx/pepijn/outputs/train/pi052_robocasa_human300_${RUN_ID}"

# --- Exclude un-annotated episodes -----------------------------------------
# 63 episodes in this dataset carry NO `subtask` annotation (no persistent
# language rows at all). `--dataset.episodes` is an INCLUDE list, so we pass
# the complement: every episode index except those 63. The helper reads
# meta/info.json from the Hub to confirm total_episodes (32043) and validates
# the excluded indices are in range before emitting the list. If the dataset
# version changes such that the indices fall out of range, the helper aborts
# the job rather than silently training on the wrong episodes.
echo "Building episode include-list (excluding un-annotated episodes)..."
EPISODES=$(python scripts/build_episode_filter.py \
    --repo-id "$DATASET" \
    --revision "$DATASET_REVISION")

echo "Training pi052 on $DATASET with ${NUM_PROCESSES} GPUs, batch size ${BATCH_SIZE}/GPU, ${STEPS} steps"
echo "Output directory: $OUTPUT_DIR"
export LEROBOT_DUMP_RECIPE_SAMPLES=8

accelerate launch --multi_gpu --num_processes="$NUM_PROCESSES" \
    -m lerobot.scripts.lerobot_train \
    --policy.type=pi052 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.recipe_path=recipes/subtask_mem_vqa_robocasa.yaml \
    --dataset.repo_id="$DATASET" \
    --dataset.revision="$DATASET_REVISION" \
    --dataset.episodes="$EPISODES" \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO_ID" \
    --policy.compile_model=true \
    --policy.compile_mode=default \
    --policy.gradient_checkpointing=true \
    --policy.device=cuda \
    --policy.tokenizer_max_length=256 \
    --policy.action_tokenizer_name=lerobot/fast-action-tokenizer \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.max_action_tokens=256 \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.optimizer_lr=5e-5 \
    --policy.optimizer_grad_clip_norm=1.0 \
    --policy.scheduler_decay_lr=5e-6 \
    --policy.lm_head_lr_scale=5.0 \
    --ema.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=hirobot \
    --log_freq=100 \
    --save_freq=5000 \
    --num_workers=4 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --dataset.image_transforms.random_order=true \
    --policy.auto_fit_fast_tokenizer=true \
    --policy.knowledge_insulation=true
