#!/usr/bin/env python

"""
Train LeRobot GR00T policies on Modal.

This script demonstrates how to train a GR00T policy on Modal using GPU resources.
It follows Modal's best practices for defining images, volumes, and remote functions.

Training automatically resumes from the last checkpoint if found, allowing you to
continue from where a previous run left off.

Setup:
    1. Edit the configuration constants below (HF_USER, DATASET_REPO_ID, POLICY_REPO_ID)
    2. Ensure you have Modal secrets set up:
       - huggingface-secret (with HF_TOKEN)
       - wandb-secret (with WANDB_API_KEY)

Usage:
    # Train with a specific dataset
    modal run lerobot_gr00t_modal_train.py --dataset-name rubix_stack
    modal run lerobot_gr00t_modal_train.py --dataset-name my_other_dataset

    # Detached mode (runs in background, no log streaming)
    modal run --detach lerobot_gr00t_modal_train.py --dataset-name rubix_stack

    # Launch multiple parallel training runs
    modal run --detach lerobot_gr00t_modal_train.py --dataset-name dataset_1
    modal run --detach lerobot_gr00t_modal_train.py --dataset-name dataset_2

    # With environment variables
    HF_USER=your-username modal run lerobot_gr00t_modal_train.py --dataset-name rubix_stack
    NUM_GPUS=4 modal run lerobot_gr00t_modal_train.py --dataset-name rubix_stack

Features:
    - Multi-GPU training with Accelerate (configurable via NUM_GPUS env var, default: 2)
    - Automatically resumes from last checkpoint if found
    - Survives network interruptions (training runs on Modal infrastructure)
    - Monitor progress via WandB dashboard and Modal dashboard
    - Optimized for small dataset fine-tuning (15k steps, batch_size=8, prevents overfitting)
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

# Training hyperparameters optimized for single dataset fine-tuning
# Based on NVIDIA's recommendations:
# - 10-15k steps prevents overfitting on small datasets
# - Frequent checkpointing (every 2k steps) captures best model before plateau
# - More frequent eval/logging helps identify when loss plateaus

# Create Modal app
app = modal.App("lerobot-groot-training")

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
def train(dataset_name: str, resume: bool = False):
    """Train a GR00T policy on Modal using lerobot-train."""
    from pathlib import Path

    # Compute dataset-specific configuration
    dataset_repo_id = f"{HF_USER}/{dataset_name}"
    policy_repo_id = f"{HF_USER}/so101_gr00t_{dataset_name}_{today_date}"
    output_dir_str = f"/outputs/train/so101_gr00t_{dataset_name}_{today_date}"
    job_name = f"so101_gr00t_{dataset_name}_{today_date}"

    # Install local lerobot in editable mode
    print("Installing local lerobot in editable mode...")
    subprocess.run(
        ["pip", "install", "-e", "/lerobot", "--no-deps"],
        check=True,
    )

    output_dir = Path(output_dir_str)

    # Check for existing checkpoint to enable automatic resume
    # Try to find the last checkpoint - either via symlink or by finding the latest numbered checkpoint
    checkpoints_dir = output_dir / "checkpoints"
    checkpoint_dir = None
    resume_from_checkpoint = resume or False

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
        "--num_machines=1",
        "--dynamo_backend=no",
        "--mixed_precision=no",
        "-m",
        "lerobot.scripts.lerobot_train",
        # Dataset configuration
        f"--dataset.repo_id={dataset_repo_id}",
        # Policy configuration
        "--policy.type=groot",
        f"--policy.repo_id={policy_repo_id}",
        "--policy.device=cuda",
        "--policy.push_to_hub=true",
        # Output configuration
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        # Checkpoint settings
        "--save_checkpoint=true",  # Explicitly enable checkpoint saving
        "--save_freq=2000",  # Save every 2k steps to capture best checkpoint
        # Training hyperparameters
        "--batch_size=8",  # 8 per GPU = 16 total with 2 GPUs (good for GR00T small dataset)
        "--optimizer.lr=1e-4",  # conservative LR for small datasets
        "--steps=15000",  # NVIDIA recommends 10-15k for single dataset fine-tuning
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
                f"--config_path={output_dir}/checkpoints/last/pretrained_model/train_config.json"
            ]
        )

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Commit volumes to persist data
    outputs_volume.commit()
    print("Training outputs committed to Modal volume")


@app.function(
    image=training_image,
    volumes={"/outputs": outputs_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1 * HOURS,
)
def upload_checkpoint(dataset_name: str, step: int = 0):
    """Upload checkpoint to HuggingFace. If step=0, uploads latest."""
    from pathlib import Path
    from huggingface_hub import HfApi

    # Compute dataset-specific configuration
    policy_repo_id = f"{HF_USER}/so101_gr00t_{dataset_name}_{today_date}"
    output_dir_str = f"/outputs/train/so101_gr00t_{dataset_name}_{today_date}"

    checkpoints_dir = Path(output_dir_str) / "checkpoints"

    # Find checkpoint
    if step == 0:
        checkpoint_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name),
            reverse=True,
        )
        checkpoint_dir = checkpoint_dirs[0] if checkpoint_dirs else None
    else:
        checkpoint_dir = checkpoints_dir / f"{step:06d}"

    if not checkpoint_dir or not checkpoint_dir.exists():
        print(f"No checkpoint found")
        return

    pretrained_dir = checkpoint_dir / "pretrained_model"
    print(f"Uploading step {checkpoint_dir.name} to {policy_repo_id}")

    api = HfApi()
    api.create_repo(repo_id=policy_repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=policy_repo_id,
        folder_path=pretrained_dir,
        commit_message=f"Checkpoint step {checkpoint_dir.name}",
    )
    print(f"Done: https://huggingface.co/{policy_repo_id}")


@app.local_entrypoint()
def main(dataset_name: str, resume: bool = False):
    """
    Local entrypoint that calls training with log streaming.

    Automatically resumes from last checkpoint if available.

    Usage:
        modal run lerobot_gr00t_modal_train.py --dataset-name rubix_stack
        modal run lerobot_gr00t_modal_train.py --dataset-name my_other_dataset
    """

    if dataset_name is None:
        raise ValueError("Dataset name is required")

    dataset_repo_id = f"{HF_USER}/{dataset_name}"
    policy_repo_id = f"{HF_USER}/tiny"
    output_dir = f"/outputs/train/so101_gr00t_{dataset_name}_{today_date}"
    job_name = f"so101_gr00t_{dataset_name}_{today_date}"

    print("🚀 Starting training job on Modal...")
    print(f"   Dataset: {dataset_repo_id}")
    print(f"   Policy output: {policy_repo_id}")
    print(f"   Job name: {job_name}")
    print(f"   Output directory: {output_dir}")
    print(f"   GPUs: {NUM_GPUS} x A100")
    print(f"   Batch size: 8 per GPU (total effective: {8 * NUM_GPUS})")
    print()
    print("✓ Training will automatically resume from last checkpoint if found")
    print()
    print("💡 Tip: For long training runs, use 'modal run --detach' to run in background")
    print("💡 Tip: To use more GPUs, set NUM_GPUS=4 modal run lerobot_modal_train.py")
    print()
    print("Monitor progress via:")
    print("  - WandB dashboard (URL will be logged below)")
    print("  - Modal dashboard: https://modal.com/apps")
    print()
    print("=" * 80)
    print()

    # Call training function - streams logs to terminal
    train.remote(dataset_name, resume)
