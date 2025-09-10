#!/usr/bin/env python

"""Script to create and push a PI0OpenPI model to HuggingFace hub with proper config format."""

import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo

from lerobot.policies.pi0_openpi import PI0OpenPIConfig, PI0OpenPIPolicy


def create_and_push_model(
    repo_id: str,
    private: bool = False,
    token: str = None,
):
    """Create a PI0OpenPI model with proper config and push to HuggingFace hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        private: Whether to create a private repository
        token: HuggingFace API token (optional, will use cached token if not provided)
    """
    print("=" * 60)
    print("PI0OpenPI Model Hub Upload")
    print("=" * 60)

    # Create configuration
    print("\nCreating PI0OpenPI configuration...")
    config = PI0OpenPIConfig(
        # Model architecture
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,  # Use PI0 (not PI0.5)
        dtype="float32",  # Use float32 for compatibility
        # Input/output dimensions
        action_dim=32,  # see openpi `Pi0Config`
        state_dim=32,
        action_horizon=50,
        n_action_steps=50,
        # Image inputs, see openpi `model.py, IMAGE_KEYS`
        image_keys=(
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ),
        # Training settings
        gradient_checkpointing=False,
        compile_model=False,
        device=None,  # Auto-detect
        # Tokenizer settings
        tokenizer_max_length=48,  # see openpi `__post_init__`, use pi0=48 and pi05=200
    )

    print(f"  - Config type: {config.__class__.__name__}")
    print(f"  - PaliGemma variant: {config.paligemma_variant}")
    print(f"  - Action expert variant: {config.action_expert_variant}")
    print(f"  - Action dim: {config.action_dim}")
    print(f"  - State dim: {config.state_dim}")

    # Create dummy dataset stats for normalization
    print("\nCreating dataset statistics...")
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(config.state_dim),
            "std": torch.ones(config.state_dim),
            "min": torch.full((config.state_dim,), -5.0),
            "max": torch.full((config.state_dim,), 5.0),
        },
        "action": {
            "mean": torch.zeros(config.action_dim),
            "std": torch.ones(config.action_dim),
            "min": torch.full((config.action_dim,), -1.0),
            "max": torch.full((config.action_dim,), 1.0),
        },
    }

    # Add image stats
    for key in config.image_keys:
        dataset_stats[key] = {
            "mean": torch.tensor([0.485, 0.456, 0.406]),  # TODO(pepijn): fix this, now its ImageNet mean
            "std": torch.tensor([0.229, 0.224, 0.225]),  # TODO(pepijn): fix this, now its ImageNet std
            "min": torch.tensor([0.0, 0.0, 0.0]),
            "max": torch.tensor([1.0, 1.0, 1.0]),
        }

    # Create the policy
    print("\nInitializing PI0OpenPI policy...")
    print("  (This may take a moment as it loads the tokenizer and initializes the model)")
    policy = PI0OpenPIPolicy(config, dataset_stats)

    # Initialize with small random weights (optional - for testing)
    # Note: In practice, you would load your trained weights here
    print("\nInitializing model weights...")
    for name, param in policy.named_parameters():
        if "weight" in name:
            if "norm" in name.lower() or "layernorm" in name.lower():
                torch.nn.init.ones_(param)
            elif len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param, gain=0.01)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
        elif "bias" in name:
            torch.nn.init.zeros_(param)

    print(f"  - Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        save_path.mkdir(exist_ok=True)

        print(f"\nSaving model to temporary directory: {save_path}")

        # Save the model using LeRobot's save_pretrained method
        # This ensures the config is saved in the correct format
        policy.save_pretrained(save_path)

        # List saved files
        saved_files = list(save_path.glob("*"))
        print("\nSaved files:")
        for file in saved_files:
            size = file.stat().st_size
            print(f"  - {file.name}: {size:,} bytes")

        # Create or get repository
        print(f"\nCreating/accessing repository: {repo_id}")
        api = HfApi(token=token)

        try:
            # Create repo if it doesn't exist
            create_repo(
                repo_id,
                private=private,
                token=token,
                exist_ok=True,
            )
            print(f"  ✓ Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"  ⚠️  Note: {e}")

        # Upload to hub
        print("\nUploading to HuggingFace hub...")
        api.upload_folder(
            folder_path=str(save_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload PI0OpenPI model with proper LeRobot config format",
        )

        print(f"\n✓ Model successfully uploaded to: https://huggingface.co/{repo_id}")

    print("\n" + "=" * 60)
    print("✓ Process complete!")
    print("=" * 60)

    return policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Push PI0OpenPI model to HuggingFace hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="test-user/pi0-openpi-test",
        help="HuggingFace repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, uses cached token if not provided)",
    )

    args = parser.parse_args()

    # Run the upload
    create_and_push_model(
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
    )
