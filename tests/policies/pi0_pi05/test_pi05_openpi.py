#!/usr/bin/env python

"""Test script to verify PI0.5 (pi05) support in PI0OpenPI policy, only meant to be run locally!"""

import os

import pytest
import torch

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local OpenPI installation and is not meant for CI",
)

from lerobot.policies.pi05_openpi import PI05OpenPIConfig, PI05OpenPIPolicy  # noqa: E402
from tests.utils import require_cuda  # noqa: E402


@require_cuda
def test_pi05_model_architecture():
    """Test that pi05=True creates the correct model architecture."""

    # Create config
    config = PI05OpenPIConfig(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
    )

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )
    assert config.discrete_state_input == True, (  # noqa: E712
        f"Expected discrete_state_input=True for pi05, got {config.discrete_state_input}"
    )

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
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05OpenPIPolicy(config, dataset_stats)

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )


@require_cuda
def test_pi05_forward_pass():
    """Test forward pass with"""

    # Create config
    config = PI05OpenPIConfig(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        chunk_size=16,  # Shorter chunk_size for testing
        n_action_steps=16,  # Shorter action steps for testing
    )

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

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
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05OpenPIPolicy(config, dataset_stats)

    # Create test batch
    batch_size = 2
    device = next(policy.parameters()).device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "task": ["Pick up the object"] * batch_size,
    }

    # Test forward pass
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() >= 0, "Loss should be non-negative"
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    # Test action prediction
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
        print(f"Action prediction successful. Action shape: {action.shape}")
        # When batch_size > 1, select_action returns (batch_size, action_dim)
        assert action.shape == (batch_size, 7), f"Expected action shape ({batch_size}, 7), got {action.shape}"
        assert not torch.isnan(action).any(), "Action contains NaN values"
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise
