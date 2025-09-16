#!/bin/bash
#SBATCH --job-name=smolvla_optimized_8gpu_fresh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/dana_aubakirova/vla/logs/smolvla_optimized_8gpu_fresh_%j.out
#SBATCH --error=/fsx/dana_aubakirova/vla/logs/smolvla_optimized_8gpu_fresh_%j.err
#SBATCH --exclusive

# Create logs directory if it doesn't exist
mkdir -p /fsx/dana_aubakirova/vla/logs

# Activate conda environment
source /fsx/dana_aubakirova/miniconda/etc/profile.d/conda.sh
conda activate lerobot

# Add local lerobot source to Python path to use development version
export PYTHONPATH="/fsx/dana_aubakirova/vla/lerobot/src:$PYTHONPATH"

# OPTIMIZED 8-GPU CUDA environment - high performance configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false
export HF_ACCELERATE_DEVICE_MAP=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# 8-GPU optimizations
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Change to working directory
cd /fsx/dana_aubakirova/vla

# FRESH START 8-GPU training configuration - NEW OUTPUT DIRECTORY
export OUTPUT_DIR="/fsx/dana_aubakirova/vla/outputs/train_smolvla_optimized_8gpu_fresh_$(date +%Y%m%d_%H%M%S)"
# Use ALL datasets from relative_datasets_list.txt - full scale training
export REPO_IDS=$(cat dataset_lists/all_datasets_relative.txt)

# Model configuration - optimized for 8-GPU with global batch size 32
export VLM_REPO_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
export STEPS=200000   # Full training steps
export BATCH_SIZE=8  # 4 per GPU = 32 global batch size (prevent hanging)
export EVAL_FREQ=-1 # Disable evaluation for faster training
export NUM_WORKERS=0  # MEMORY FIX: Disable workers to prevent memory exhaustion
export SAVE_FREQ=10000 # Save every 10k steps

# Model config - optimized settings inspired by SmolPi0
export POLICY=smolvla2
export USE_AMP=false   # DISABLE AMP for stability
export OPTIMIZER_LR=5e-4  # Optimized learning rate
export PEFT_METHOD=lora
export LOAD_VLM_WEIGHTS=true
export MAX_ACTION_DIM=32
export MAX_STATE_DIM=32

# Dataset config - optimized from analysis
export USE_IMAGENET_STATS=false
export ENABLE_IMG_TRANSFORM=true
export MAX_NUM_IMAGES=2  # OPTIMIZED: 2 images for better context
export MAX_IMAGE_DIM=256 # OPTIMIZED: 256px resolution
export TRAIN_ON_ALL_FEATURES=true
export FEATURES_VERSION=2

# Advanced optimizations for 8-GPU setup
export FPS_MIN=30
export FPS_MAX=30
export GRADIENT_ACCUMULATION_STEPS=1  # Global batch size = 4 × 8 × 2 = 64
export PRECISION=no
export DROP_LAST=true

# SmolPi0-inspired VLM parameters
export VLM_LAYERS=16
export EXPERT_WIDTH_MULTIPLIER=0.75
export CAUSAL_ACTION_ATTENTION=true
export SELF_ATTN_EVERY_N_LAYERS=2
export ATTENTION_MODE=cross_attn
export LORA_R=32
export LORA_TARGET_MODULES=q_proj,v_proj
export PREFIX_LENGTH=0

# Learning rate schedule inspired by SmolPi0
export DECAY_LR=1e-6
export DECAY_STEPS=50000
export LR_VLM=1e-4
export WARMUP_STEPS=1000

# Set environment variables for model cache and offline mode
export HF_LEROBOT_HOME="/fsx/dana_aubakirova/vla"
export HF_HOME="/fsx/dana_aubakirova/vla/.cache/huggingface"
export HF_HUB_CACHE="/fsx/dana_aubakirova/vla/.cache/huggingface"
export TRANSFORMERS_CACHE="/fsx/dana_aubakirova/vla/.cache/huggingface"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Optimized accelerate config
export ACCELERATE_CONFIG_FILE="/fsx/dana_aubakirova/vla/accelerate_configs/optimized_fresh_config.yaml"

# Wandb configuration - FRESH START
export WANDB_PROJECT="smolvla2-training"
export WANDB_NOTES="8-GPU optimized training FRESH START - same parameters as previous run but from scratch"
export WANDB_MODE="online"

# Print comprehensive optimization info
echo "🚀 =============================================="
echo "🚀 OPTIMIZED 8-GPU FRESH START TRAINING"
echo "🚀 =============================================="
echo "🆕 FRESH START - No resume, new output directory"
echo "📊 Datasets: ALL available datasets (same as previous run)"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🎯 Policy: $POLICY"
echo "🔧 Batch size per GPU: $BATCH_SIZE (GLOBAL BATCH SIZE: $((BATCH_SIZE * 8)))"
echo "🔄 Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "📈 Training steps: $STEPS"
echo "💾 Save frequency: $SAVE_FREQ"
echo "🔬 Evaluation frequency: $EVAL_FREQ"
echo "⚡ AMP enabled: $USE_AMP (no mixed precision - stable)"
echo "📚 Learning rate: $OPTIMIZER_LR"
echo "🎓 VLM Learning rate: $LR_VLM"
echo "🔥 Warmup steps: $WARMUP_STEPS"
echo "📷 Max images: $MAX_NUM_IMAGES"
echo "🖼️  Image dimension: $MAX_IMAGE_DIM"
echo "👥 Data workers per GPU: $NUM_WORKERS (memory optimized)"
echo "🧠 VLM layers: $VLM_LAYERS"
echo "🔄 Expert width multiplier: $EXPERT_WIDTH_MULTIPLIER"
echo "🎯 LORA rank: $LORA_R"
echo "🖥️  GPUs: 8 (HIGH PERFORMANCE)"
echo "📊 Wandb project: $WANDB_PROJECT"
echo "🚀 =============================================="

# Check GPU availability
echo "🖥️  GPU Information:"
nvidia-smi --list-gpus

# Create optimized 8-GPU accelerate config
mkdir -p /fsx/dana_aubakirova/vla/accelerate_configs
cat > /fsx/dana_aubakirova/vla/accelerate_configs/optimized_fresh_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: '0,1,2,3,4,5,6,7'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "📋 Created optimized accelerate config with no mixed precision for stability"

# Run distributed training with optimized accelerate configuration - FRESH START
accelerate launch --config_file /fsx/dana_aubakirova/vla/accelerate_configs/optimized_fresh_config.yaml \
    lerobot/src/lerobot/scripts/train.py \
    --policy.type=$POLICY \
    --dataset.repo_id="$REPO_IDS" \
    --dataset.root="/fsx/dana_aubakirova/vla" \
    --dataset.use_imagenet_stats=$USE_IMAGENET_STATS \
    --dataset.image_transforms.enable=$ENABLE_IMG_TRANSFORM \
    --dataset.train_on_all_features=$TRAIN_ON_ALL_FEATURES \
    --dataset.features_version=$FEATURES_VERSION \
    --policy.max_action_dim=$MAX_ACTION_DIM \
    --policy.max_state_dim=$MAX_STATE_DIM \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --policy.use_amp=$USE_AMP \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.optimizer_lr_vlm=$LR_VLM \
    --policy.scheduler_decay_lr=$DECAY_LR \
    --policy.scheduler_decay_steps=$DECAY_STEPS \
    --policy.scheduler_warmup_steps=$WARMUP_STEPS \
    --policy.peft_method=$PEFT_METHOD \
    --policy.peft_config.r=$LORA_R \
    --policy.peft_config.target_modules=$LORA_TARGET_MODULES \
    --policy.load_vlm_weights=$LOAD_VLM_WEIGHTS \
    --policy.repo_id=$VLM_REPO_ID \
    --policy.push_to_hub=false \
    --dataset.max_num_images=$MAX_NUM_IMAGES \
    --dataset.max_image_dim=$MAX_IMAGE_DIM \
    --dataset.video_backend=pyav \
    --num_workers=$NUM_WORKERS \
    --wandb.enable=true \
    --wandb.project=$WANDB_PROJECT \
    --wandb.notes="$WANDB_NOTES" \
    --dataset.min_fps=$FPS_MIN \
    --dataset.max_fps=$FPS_MAX \
    --policy.num_vlm_layers=$VLM_LAYERS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH_MULTIPLIER \
    --policy.causal_action_attention_mask=$CAUSAL_ACTION_ATTENTION \
    --policy.self_attn_every_n_layers=$SELF_ATTN_EVERY_N_LAYERS \
    --policy.attention_mode=$ATTENTION_MODE \
    --policy.prefix_length=$PREFIX_LENGTH

echo "✅ Optimized 8-GPU FRESH START training completed! Check results in: $OUTPUT_DIR"
echo "📊 View training progress at: https://wandb.ai"
echo "🆕 FRESH START TRAINING SUMMARY:"
echo "   • Started from scratch with new output directory"
echo "   • Training from step 0 to step $STEPS"
echo "   • Same optimized parameters as previous successful run"
echo "   • New WandB run will be created automatically"
echo ""
echo "🚀 Key 8-GPU optimizations applied:"
echo "   • 8 GPUs with global batch size $((BATCH_SIZE * 8))"
echo "   • Memory-optimized data loading: 0 workers (prevents OOM)"
echo "   • STABLE: No mixed precision (matches conservative setup)"
echo "   • Optimized NCCL settings for 8-GPU communication"
echo "   • Enhanced memory allocation for high-throughput"
echo ""
echo "🏃‍♂️ Expected performance gains:"
echo "   • ~4x faster training throughput vs single GPU"
echo "   • Clean start without any checkpoint compatibility issues"
echo "   • Proven parameter configuration from previous run"
