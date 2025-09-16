#!/usr/bin/env python

"""Test script to verify PI0.5 (pi05) support in PI0OpenPI policy."""

import pytest
import torch

# Skip entire module if transformers is not available
pytest.importorskip("transformers")

from lerobot.policies.pi0_openpi.configuration_pi0openpi import PI0OpenPIConfig
from lerobot.policies.pi0_openpi.modeling_pi0openpi import PI0OpenPIPolicy
from lerobot.policies.pi05_openpi import PI05OpenPIConfig, PI05OpenPIPolicy
from tests.utils import require_nightly_gpu


@require_nightly_gpu
def test_pi05_model_architecture():
    """Test that pi05=True creates the correct model architecture."""
    print("Testing PI0.5 model architecture...")

    # Create config
    config = PI05OpenPIConfig(
        action_dim=7,
        state_dim=14,
        dtype="float32",
    )

    # Verify tokenizer max length is set correctly
    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )
    print(f"✓ Tokenizer max length correctly set to {config.tokenizer_max_length}")

    # Verify discrete_state_input defaults to pi05
    assert config.discrete_state_input == True, (  # noqa: E712
        f"Expected discrete_state_input=True for pi05, got {config.discrete_state_input}"
    )
    print(f"✓ discrete_state_input correctly defaults to pi05 value: {config.discrete_state_input}")

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
    policy = PI05OpenPIPolicy(config, dataset_stats)

    # Verify pi05 model components exist

    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"
    print("✓ Time MLP layers present for AdaRMS conditioning")

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )
    print("✓ Action-time MLP layers correctly absent")

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"
    print("✓ State projection layer correctly absent")

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )
    print("✓ AdaRMS correctly configured: PaliGemma=False, Expert=True")


@require_nightly_gpu
def test_pi05_forward_pass():
    """Test forward pass with"""
    print("\nTesting PI0.5 forward pass...")

    # Create config
    config = PI05OpenPIConfig(
        action_dim=7,
        state_dim=14,
        dtype="float32",
        chunk_size=16,  # Shorter chunk_size for testing
        n_action_steps=16,  # Shorter action steps for testing
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
        print(f"✓ Forward pass successful. Loss: {loss_dict['loss']:.4f}")
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() >= 0, "Loss should be non-negative"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise

    # Test action prediction
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
        print(f"✓ Action prediction successful. Action shape: {action.shape}")
        # When batch_size > 1, select_action returns (batch_size, action_dim)
        assert action.shape == (batch_size, 7), f"Expected action shape ({batch_size}, 7), got {action.shape}"
        assert not torch.isnan(action).any(), "Action contains NaN values"
    except Exception as e:
        print(f"✗ Action prediction failed: {e}")
        raise


@require_nightly_gpu
def test_pi0_vs_pi05_differences():
    """Test key differences between pi0 and pi05 modes."""
    print("\nComparing PI0 vs PI0.5 architectures...")

    # Create both configurations
    config_pi0 = PI0OpenPIConfig(action_dim=7, state_dim=14, dtype="float32")
    config_pi05 = PI05OpenPIConfig(action_dim=7, state_dim=14, dtype="float32")

    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }

    # Create both models
    policy_pi0 = PI0OpenPIPolicy(config_pi0, dataset_stats)
    policy_pi05 = PI05OpenPIPolicy(config_pi05, dataset_stats)

    print("\nPI0 Model:")
    print(f"  - Tokenizer max length: {config_pi0.tokenizer_max_length}")
    print(f"  - Has state_proj: {hasattr(policy_pi0.model, 'state_proj')}")
    print(f"  - Has action_time_mlp: {hasattr(policy_pi0.model, 'action_time_mlp_in')}")
    print(f"  - Has time_mlp: {hasattr(policy_pi0.model, 'time_mlp_in')}")
    print(f"  - Uses AdaRMS: {policy_pi0.model.paligemma_with_expert.gemma_expert.config.use_adarms}")

    print("\nPI0.5 Model:")
    print(f"  - Tokenizer max length: {config_pi05.tokenizer_max_length}")
    print(f"  - discrete_state_input: {config_pi05.discrete_state_input}")
    print(f"  - Has state_proj: {hasattr(policy_pi05.model, 'state_proj')}")
    print(f"  - Has action_time_mlp: {hasattr(policy_pi05.model, 'action_time_mlp_in')}")
    print(f"  - Has time_mlp: {hasattr(policy_pi05.model, 'time_mlp_in')}")
    print(f"  - Uses AdaRMS: {policy_pi05.model.paligemma_with_expert.gemma_expert.config.use_adarms}")

    # Count parameters
    pi0_params = sum(p.numel() for p in policy_pi0.parameters())
    pi05_params = sum(p.numel() for p in policy_pi05.parameters())

    print("\nParameter counts:")
    print(f"  - PI0: {pi0_params:,}")
    print(f"  - PI0.5: {pi05_params:,}")
    print(f"  - Difference: {pi0_params - pi05_params:,} (PI0.5 has fewer params due to no state embedding)")
