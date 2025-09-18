#!/usr/bin/env python

"""Test script to verify PI0OpenPI policy integration with LeRobot."""

import pytest
import torch

# Skip entire module if transformers is not available
pytest.importorskip("transformers")

from lerobot.policies.factory import make_policy_config
from lerobot.policies.pi0_openpi import PI0OpenPIConfig, PI0OpenPIPolicy
from tests.utils import require_nightly_gpu


@require_nightly_gpu
def test_policy_instantiation():
    """Test basic policy instantiation."""
    print("Testing PI0OpenPI policy instantiation...")

    # Create config
    config = PI0OpenPIConfig(action_dim=7, state_dim=14, dtype="float32")

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
    }

    # Instantiate policy
    policy = PI0OpenPIPolicy(config, dataset_stats)
    print(f"Policy created successfully: {policy.name}")

    # Test forward pass with dummy data
    batch_size = 1
    device = policy.device if hasattr(policy, "device") else "cpu"
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }

    print("\nTesting forward pass...")
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"✓ Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise

    print("\nTesting action prediction...")
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
        print(f"✓ Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"✗ Action prediction failed: {e}")
        raise


@require_nightly_gpu
def test_config_creation():
    """Test policy config creation through factory."""
    print("\nTesting config creation through factory...")

    try:
        config = make_policy_config(
            policy_type="pi0_openpi",
            action_dim=7,
            state_dim=14,
        )
        print("✓ Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        raise
