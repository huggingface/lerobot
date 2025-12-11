#!/usr/bin/env python

"""
Train LeRobot Pi 0.5 policies on Modal.

This script demonstrates how to train a Pi 0.5 policy on Modal using GPU resources.
It follows Modal's best practices for defining images, volumes, and remote functions.

Training automatically resumes from the last checkpoint if found, allowing you to
continue from where a previous run left off.

Setup:
    1. Edit the configuration constants below (HF_USER, DATASET_REPO_ID, POLICY_REPO_ID)
    2. Ensure you have Modal secrets set up:
       - huggingface-secret (with HF_TOKEN)
       - wandb-secret (with WANDB_API_KEY)

Usage:
    # Attached mode (streams logs, can disconnect with Ctrl+C)
    modal run lerobot_pi_modal_train.py

    # Detached mode (runs in background, no log streaming)
    modal run --detach lerobot_pi_modal_train.py

    # With environment variables
    HF_USER=your-username modal run lerobot_pi_modal_train.py
    NUM_GPUS=4 modal run lerobot_pi_modal_train.py  # Use 4 GPUs instead of default 2
    HF_USER=myuser NUM_GPUS=2 modal run lerobot_pi_modal_train.py

Features:
    - Multi-GPU training with Accelerate (configurable via NUM_GPUS env var, default: 2)
    - Automatically resumes from last checkpoint if found
    - Survives network interruptions (training runs on Modal infrastructure)
    - Monitor progress via WandB dashboard and Modal dashboard
    - Optimized with Pi 0.5 recommended settings (compile, gradient checkpointing, bfloat16)
    - Frequent checkpointing every 2k steps to capture best model
"""

import os
import subprocess
from datetime import datetime

import modal

# Modal best practices: define constants
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# Training configuration - MODIFY THESE VALUES

today_date = datetime.now().strftime("%Y-%m-%d")

HF_USER = os.environ.get("HF_USER", "lbxa")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))  # Number of GPUs for multi-GPU training
DATASET_NAME = "rubix_stack_v2"
DATASET_REPO_ID = f"{HF_USER}/{DATASET_NAME}"
POLICY_REPO_ID = f"{HF_USER}/pi05_{DATASET_NAME}"
OUTPUT_DIR = f"/outputs/train/so101_pi05_{DATASET_NAME}_{today_date}"
JOB_NAME = f"so101_pi05_{DATASET_NAME}_{today_date}"

# Training hyperparameters optimized for Pi 0.5 fine-tuning
# Based on Physical Intelligence recommendations:
# - Model compilation for faster training
# - Gradient checkpointing for memory efficiency
# - bfloat16 for mixed precision training
# - Frequent checkpointing (every 2k steps) captures best model before plateau

# Create Modal app
app = modal.App("lerobot-pi05-training")

# Define volumes for datasets and outputs
datasets_volume = modal.Volume.from_name("lerobot-datasets", create_if_missing=True)
outputs_volume = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)

# Build container image with CUDA support
cuda_version = "12.8.1"  # Must be <= Modal's host CUDA version
flavor = "devel"  # Includes full CUDA toolkit (nvcc, etc.)
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
torch_cuda_arch_list = "8.0"  # Target NVIDIA A100 (SM80)

training_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .entrypoint([])  # Remove verbose logging by base image
    .apt_install(
        "git",
        "build-essential",
        "pkg-config",
        "cmake",
        "ninja-build",
        "clang",
        # FFmpeg libraries for video decoding (required by torchcodec/pyav)
        "ffmpeg",
        "libavcodec-dev",
        "libavformat-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libavfilter-dev",
    )
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
    )
    .uv_pip_install(
        "torch>=2.2.1,<2.8.0",
        "torchvision>=0.21.0,<0.23.0",
        "ninja",
        "packaging>=24.2,<26.0",
        "psutil",
    )
    .run_commands(
        f"TORCH_CUDA_ARCH_LIST={torch_cuda_arch_list} pip install --no-build-isolation 'flash-attn>=2.5.9,<3.0.0'",
        "python -c \"import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')\"",
    )
    .uv_pip_install(
        "accelerate>=1.10.0",
        "transformers>=4.53.0",
        "datasets>=4.0.0",
        "huggingface-hub[hf-transfer,cli]>=0.34.2",
        "wandb>=0.20.0",
        "termcolor>=2.4.0",
        "einops>=0.8.0",
        "peft>=0.13.0",
        "opencv-python-headless>=4.9.0",
        "av>=15.0.0",
        "jsonlines>=4.0.0",
        "pyserial>=3.5",
        "draccus==0.10.0",
        "gymnasium>=1.1.1",
        "rerun-sdk>=0.24.0",
        "deepdiff>=7.0.1",
        "imageio[ffmpeg]>=2.34.0",
        "diffusers>=0.27.0",
        "dm-tree>=0.1.8",
        "timm>=1.0.0",
        "safetensors>=0.4.3",
        "Pillow>=10.0.0",
        # Pi 0.5 specific dependencies
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    )
    .add_local_dir(
        ".", remote_path="/lerobot", copy=False
    )  # Mount lerobot directory (MUST BE LAST for dev workflow)
)


@app.function(
    gpu=f"A100:{NUM_GPUS}",  # Multi-GPU support (configurable via NUM_GPUS env var)
    image=training_image,
    volumes={
        "/datasets": datasets_volume,
        "/outputs": outputs_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # Contains HF_TOKEN
        modal.Secret.from_name("wandb-secret"),  # Contains WANDB_API_KEY
    ],
    timeout=24 * HOURS,
)
def train():
    """Train a Pi 0.5 policy on Modal using lerobot-train."""
    from pathlib import Path

    # Install local lerobot in editable mode with pi dependencies
    print("Installing local lerobot in editable mode with pi dependencies...")
    subprocess.run(
        ["pip", "install", "-e", "/lerobot[pi]"],
        check=True,
    )

    output_dir = Path(OUTPUT_DIR)

    # Check for existing checkpoint to enable automatic resume
    # Try to find the last checkpoint - either via symlink or by finding the latest numbered checkpoint
    checkpoints_dir = output_dir / "checkpoints"
    checkpoint_dir = None
    resume_from_checkpoint = False

    if checkpoints_dir.exists():
        # First try the "last" symlink
        last_checkpoint = checkpoints_dir / "last"
        if last_checkpoint.exists():
            checkpoint_dir = last_checkpoint
            resume_from_checkpoint = True
            print(f"✓ Found checkpoint symlink: {checkpoint_dir}")
        else:
            # If no symlink, find the highest numbered checkpoint directory
            checkpoint_dirs = sorted(
                [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda x: int(x.name),
                reverse=True,
            )
            if checkpoint_dirs:
                checkpoint_dir = checkpoint_dirs[0]
                resume_from_checkpoint = True
                print(f"✓ Found latest checkpoint: {checkpoint_dir} (step {checkpoint_dir.name})")
                if len(checkpoint_dirs) > 1:
                    print(f"  Available checkpoints: {', '.join([d.name for d in checkpoint_dirs[:5]])}")

    if resume_from_checkpoint:
        print("  Resuming training from last checkpoint...")
    else:
        if output_dir.exists():
            print(f"⚠️  Output directory exists but no valid checkpoint found: {output_dir}")
            print("   Removing existing directory to start fresh...")
            import shutil

            shutil.rmtree(output_dir)
        print("Starting training from scratch")

    # Build training command using accelerate for multi-GPU support
    cmd = [
        # Accelerate launcher configuration
        "accelerate",
        "launch",
        "--multi_gpu",
        f"--num_processes={NUM_GPUS}",
        "-m",
        "lerobot.scripts.lerobot_train",
        # Dataset configuration
        f"--dataset.repo_id={DATASET_REPO_ID}",
        "--dataset.video_backend=pyav",  # Use pyav backend for more stable video decoding
        # Policy configuration - Pi 0.5 specific
        "--policy.type=pi05",
        f"--policy.repo_id={POLICY_REPO_ID}",
        "--policy.pretrained_path=lerobot/pi05_base",
        "--policy.device=cuda",
        "--policy.push_to_hub=true",
        # Pi 0.5 optimization settings
        "--policy.compile_model=true",  # Enables model compilation for faster training
        "--policy.gradient_checkpointing=true",  # Reduces memory usage
        # Normalization mapping (using flag approach instead of quantiles)
        '--policy.normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
        # Output configuration
        f"--output_dir={output_dir}",
        f"--job_name={JOB_NAME}",
        # Checkpoint settings
        "--save_checkpoint=true",  # Explicitly enable checkpoint saving
        "--save_freq=2000",  # Save every 2k steps to capture best checkpoint
        # Training hyperparameters
        "--batch_size=8",  # Recommended batch size for Pi 0.5
        "--optimizer.lr=1e-4",  # Conservative learning rate
        "--steps=15000",  # Training steps for fine-tuning
        # Evaluation and logging settings
        "--eval_freq=2000",  # Evaluate frequently to catch plateaus early
        "--log_freq=100",  # More frequent logging for better monitoring
        # WandB settings
        "--wandb.enable=true",
    ]

    # Add resume flags if checkpoint exists
    if resume_from_checkpoint:
        cmd.extend(
            [
                "--resume=true",
                f"--checkpoint_path={checkpoint_dir}",
            ]
        )

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Commit volumes to persist data
    outputs_volume.commit()
    print("Training outputs committed to Modal volume")


@app.local_entrypoint()
def main():
    """
    Local entrypoint that calls training with log streaming.

    Automatically resumes from last checkpoint if available.
    """
    print("🚀 Starting Pi 0.5 training job on Modal...")
    print(f"   Dataset: {DATASET_REPO_ID}")
    print(f"   Policy output: {POLICY_REPO_ID}")
    print("   Pretrained model: lerobot/pi05_base")
    print(f"   Job name: {JOB_NAME}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   GPUs: {NUM_GPUS} x A100")
    print(f"   Batch size: 8 per GPU (total effective: {8 * NUM_GPUS})")
    print("   Optimizations: compile_model=true, gradient_checkpointing=true, dtype=bfloat16")
    print()
    print("✓ Training will automatically resume from last checkpoint if found")
    print()
    print("💡 Tip: For long training runs, use 'modal run --detach' to run in background")
    print("💡 Tip: To use more GPUs, set NUM_GPUS=4 modal run lerobot_pi_modal_train.py")
    print()
    print("Monitor progress via:")
    print("  - WandB dashboard (URL will be logged below)")
    print("  - Modal dashboard: https://modal.com/apps")
    print()
    print("=" * 80)
    print()

    # Call training function - streams logs to terminal
    train.remote()
