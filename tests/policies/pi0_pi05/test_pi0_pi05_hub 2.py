#!/usr/bin/env python

# TODO(pepijn): Remove these tests before merging

"""Test script to load PI0OpenPI model from HuggingFace hub and run inference."""

import os

import pytest
import torch

# Skip entire module if transformers is not available
pytest.importorskip("transformers")

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires HuggingFace authentication and is not meant for CI",
)

from lerobot.policies.pi0 import PI0Policy  # noqa: E402
from lerobot.policies.pi05.modeling_pi05openpi import PI05Policy  # noqa: E402


def create_dummy_stats(config):
    """Create dummy dataset statistics for testing."""
    dummy_stats = {
        "observation.state": {
            "mean": torch.zeros(config.max_state_dim),
            "std": torch.ones(config.max_state_dim),
        },
        "action": {
            "mean": torch.zeros(config.max_action_dim),
            "std": torch.ones(config.max_action_dim),
        },
    }

    # Add stats for image keys if they exist
    for key in config.image_features.keys():
        dummy_stats[key] = {
            "mean": torch.zeros(3, config.image_resolution[0], config.image_resolution[1]),
            "std": torch.ones(3, config.image_resolution[0], config.image_resolution[1]),
        }

    return dummy_stats


# Test data for all 6 base models
MODEL_TEST_PARAMS = [
    # PI0 models
    ("pepijn223/pi0_base_fp32", "PI0", PI0Policy),
    ("pepijn223/pi0_droid_fp32", "PI0", PI0Policy),
    ("pepijn223/pi0_libero_fp32", "PI0", PI0Policy),
    # PI0.5 models
    ("pepijn223/pi05_base_fp32", "PI0.5", PI05Policy),
    ("pepijn223/pi05_droid_fp32", "PI0.5", PI05Policy),
    ("pepijn223/pi05_libero_fp32", "PI0.5", PI05Policy),
]


@pytest.mark.parametrize("model_id,model_type,policy_class", MODEL_TEST_PARAMS)
def test_all_base_models_hub_loading(model_id, model_type, policy_class):
    """Test loading and basic functionality of all 6 base models from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "pepijn223/pi0_base_fp32")
        model_type: Model type ("PI0" or "PI0.5")
        policy_class: Policy class to use (PI0Policy or PI05Policy)
    """
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type} model: {model_id}")
    print(f"{'=' * 80}")

    # Load the model from HuggingFace hub
    try:
        policy = policy_class.from_pretrained(model_id, strict=True)
        print(f"✓ Successfully loaded {model_type} model from {model_id}")
    except Exception as e:
        print(f"✗ Failed to load model {model_id}: {e}")
        raise

    # Set up input_features and output_features in the config (not set by from_pretrained)
    from lerobot.configs.types import FeatureType, PolicyFeature

    policy.config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(policy.config.max_state_dim,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
        "observation.images.left_wrist_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
        "observation.images.right_wrist_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    policy.config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(policy.config.max_action_dim,),
        ),
    }

    # Get model info
    device = next(policy.parameters()).device
    print("\nModel configuration:")
    print(f"  - Model ID: {model_id}")
    print(f"  - Model type: {model_type}")
    print(f"  - PaliGemma variant: {policy.config.paligemma_variant}")
    print(f"  - Action expert variant: {policy.config.action_expert_variant}")
    print(f"  - Action dimension: {policy.config.max_action_dim}")
    print(f"  - State dimension: {policy.config.max_state_dim}")
    print(f"  - Chunk size: {policy.config.chunk_size}")
    print(f"  - Tokenizer max length: {policy.config.tokenizer_max_length}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {next(policy.parameters()).dtype}")

    # Verify model-specific architecture
    if model_type == "PI0.5":
        print(f"  - discrete_state_input: {policy.config.discrete_state_input}")
        # Verify PI0.5 specific features
        assert hasattr(policy.model, "time_mlp_in"), f"{model_id}: PI0.5 should have time_mlp_in"
        assert hasattr(policy.model, "time_mlp_out"), f"{model_id}: PI0.5 should have time_mlp_out"
        assert not hasattr(policy.model, "state_proj"), f"{model_id}: PI0.5 should not have state_proj"
        assert not hasattr(policy.model, "action_time_mlp_in"), (
            f"{model_id}: PI0.5 should not have action_time_mlp_in"
        )
        adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
        assert adarms_expert_config == True, f"{model_id}: PI0.5 expert should use AdaRMS"  # noqa: E712
        print("  ✓ PI0.5 architecture verified")
    else:
        # Verify PI0 specific features
        assert hasattr(policy.model, "action_time_mlp_in"), f"{model_id}: PI0 should have action_time_mlp_in"
        assert hasattr(policy.model, "action_time_mlp_out"), (
            f"{model_id}: PI0 should have action_time_mlp_out"
        )
        assert hasattr(policy.model, "state_proj"), f"{model_id}: PI0 should have state_proj"
        assert not hasattr(policy.model, "time_mlp_in"), f"{model_id}: PI0 should not have time_mlp_in"
        adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
        assert adarms_expert_config == False, f"{model_id}: PI0 expert should not use AdaRMS"  # noqa: E712
        print("  ✓ PI0 architecture verified")

    # Create dummy stats for testing
    dummy_stats = create_dummy_stats(policy.config)
    for key, stats in dummy_stats.items():
        dummy_stats[key] = {
            "mean": stats["mean"].to(device),
            "std": stats["std"].to(device),
        }

    # Initialize normalization layers with dummy stats
    from lerobot.policies.normalize import Normalize, Unnormalize

    policy.normalize_inputs = Normalize(
        policy.config.input_features, policy.config.normalization_mapping, dummy_stats
    )
    policy.normalize_targets = Normalize(
        policy.config.output_features, policy.config.normalization_mapping, dummy_stats
    )
    policy.unnormalize_outputs = Unnormalize(
        policy.config.output_features, policy.config.normalization_mapping, dummy_stats
    )

    # Create test batch
    batch_size = 1
    batch = {
        "observation.state": torch.randn(
            batch_size, policy.config.max_state_dim, dtype=torch.float32, device=device
        ),
        "action": torch.randn(
            batch_size,
            policy.config.chunk_size,
            policy.config.max_action_dim,
            dtype=torch.float32,
            device=device,
        ),
        "task": ["Pick up the object"] * batch_size,
    }

    # Add images based on config
    for key in policy.config.image_features.keys():
        batch[key] = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device)

    # Test forward pass
    print(f"\nTesting forward pass for {model_id}...")
    try:
        policy.train()
        loss, loss_dict = policy.forward(batch)
        assert not torch.isnan(loss), f"{model_id}: Forward pass produced NaN loss"
        assert loss.item() >= 0, f"{model_id}: Loss should be non-negative"
        print(f"✓ Forward pass successful - Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed for {model_id}: {e}")
        raise

    # Test action prediction
    print(f"Testing action prediction for {model_id}...")
    try:
        policy.eval()
        with torch.no_grad():
            action = policy.select_action(batch)
        expected_shape = (batch_size, policy.config.max_action_dim)
        assert action.shape == expected_shape, (
            f"{model_id}: Expected action shape {expected_shape}, got {action.shape}"
        )
        assert not torch.isnan(action).any(), f"{model_id}: Action contains NaN values"
        print(f"✓ Action prediction successful - Shape: {action.shape}")
    except Exception as e:
        print(f"✗ Action prediction failed for {model_id}: {e}")
        raise

    print(f"All tests passed for {model_id}!")
