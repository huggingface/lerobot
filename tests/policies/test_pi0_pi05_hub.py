#!/usr/bin/env python

"""Test script to load PI0OpenPI model from HuggingFace hub and run inference."""

import torch

from lerobot.policies.pi0_openpi import PI0OpenPIPolicy
from lerobot.policies.pi05_openpi.modeling_pi05openpi import PI05OpenPIPolicy
from tests.utils import require_nightly_gpu


def create_dummy_stats(config):
    """Create dummy dataset statistics for testing."""
    dummy_stats = {
        "observation.state": {
            "mean": torch.zeros(config.state_dim),
            "std": torch.ones(config.state_dim),
        },
        "action": {
            "mean": torch.zeros(config.action_dim),
            "std": torch.ones(config.action_dim),
        },
    }

    # Add stats for image keys if they exist
    for key in config.image_keys:
        dummy_stats[key] = {
            "mean": torch.zeros(3, config.image_resolution[0], config.image_resolution[1]),
            "std": torch.ones(3, config.image_resolution[0], config.image_resolution[1]),
        }

    return dummy_stats


@require_nightly_gpu
def test_pi0_hub_loading():
    """Test loading PI0 model from HuggingFace hub."""
    _test_hub_loading(model_id="pepijn223/pi0_base_fp32", model_name="PI0")


@require_nightly_gpu
def test_pi05_hub_loading():
    """Test loading PI0.5 model from HuggingFace hub."""
    _test_hub_loading(model_id="pepijn223/pi05_base_fp32", model_name="PI0.5")


def _test_hub_loading(model_id, model_name):
    """Internal helper function for testing hub loading.

    Args:
        model_id: HuggingFace model ID to load
        model_name: Display name for the model (e.g., "PI0", "PI0.5")
    """
    print("=" * 60)
    print(f"{model_name} OpenPI HuggingFace Hub Loading Test")
    print("=" * 60)

    print(f"\nLoading model from: {model_id}")
    print("-" * 60)

    try:
        # Load the model from HuggingFace hub with strict mode
        if model_name == "PI0.5":
            policy = PI05OpenPIPolicy.from_pretrained(
                model_id,
                strict=True,  # Ensure all weights are loaded correctly,
            )
        else:
            policy = PI0OpenPIPolicy.from_pretrained(
                model_id,
                strict=True,  # Ensure all weights are loaded correctly,
            )

        print("✓ Model loaded successfully from HuggingFace hub")

        # Inject dummy stats since they aren't loaded from the hub
        print("Creating dummy dataset stats for testing...")
        device = next(policy.parameters()).device
        dummy_stats = create_dummy_stats(policy.config)

        # Move dummy stats to device
        for key, stats in dummy_stats.items():
            dummy_stats[key] = {
                "mean": stats["mean"].to(device),
                "std": stats["std"].to(device),
            }

        # Initialize normalization layers with dummy stats if they have NaN/inf values
        print("✓ Dummy stats created and moved to device")

        # Get model info
        print("\nModel configuration:")
        print(f"  - Model type: {model_name}")
        print(f"  - PaliGemma variant: {policy.config.paligemma_variant}")
        print(f"  - Action expert variant: {policy.config.action_expert_variant}")
        print(f"  - Action dimension: {policy.config.action_dim}")
        print(f"  - State dimension: {policy.config.state_dim}")
        print(f"  - Chunk_size: {policy.config.chunk_size}")
        print(f"  - Tokenizer max length: {policy.config.tokenizer_max_length}")
        if model_name == "PI0.5":
            print(f"  - discrete_state_input: {policy.config.discrete_state_input}")
        print(f"  - Device: {device}")
        print(f"  - Dtype: {next(policy.parameters()).dtype}")

        # Check model-specific features
        if model_name == "PI0.5":
            print("\nPI0.5 specific features:")
            print(f"  - Has time_mlp layers: {hasattr(policy.model, 'time_mlp_in')}")
            print(f"  - Has state_proj: {hasattr(policy.model, 'state_proj')} (should be False)")
            print(f"  - Uses AdaRMS: {policy.model.paligemma_with_expert.gemma_expert.config.use_adarms}")

            # Verify PI0.5 architecture
            assert hasattr(policy.model, "time_mlp_in"), "PI0.5 should have time_mlp_in"
            assert hasattr(policy.model, "time_mlp_out"), "PI0.5 should have time_mlp_out"
            assert not hasattr(policy.model, "state_proj"), "PI0.5 should not have state_proj"
            assert not hasattr(policy.model, "action_time_mlp_in"), "PI0.5 should not have action_time_mlp_in"
            print("  ✓ PI0.5 architecture verified")
        else:
            print("\nPI0 specific features:")
            print(f"  - Has action_time_mlp layers: {hasattr(policy.model, 'action_time_mlp_in')}")
            print(f"  - Has state_proj: {hasattr(policy.model, 'state_proj')} (should be True)")
            print(
                f"  - Uses AdaRMS: {policy.model.paligemma_with_expert.gemma_expert.config.use_adarms} (should be False)"
            )

            # Verify PI0 architecture
            assert hasattr(policy.model, "action_time_mlp_in"), "PI0 should have action_time_mlp_in"
            assert hasattr(policy.model, "action_time_mlp_out"), "PI0 should have action_time_mlp_out"
            assert hasattr(policy.model, "state_proj"), "PI0 should have state_proj"
            assert not hasattr(policy.model, "time_mlp_in"), "PI0 should not have time_mlp_in"
            print("  ✓ PI0 architecture verified")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise

    print("\n" + "-" * 60)
    print("Testing forward pass with loaded model...")

    # Create dummy batch for testing
    batch_size = 1

    # Check if normalization layers have invalid stats and replace with dummy stats if needed
    try:
        # Check if the normalize_inputs has valid stats
        if hasattr(policy.normalize_inputs, "stats"):
            obs_state_mean = policy.normalize_inputs.stats.get("observation.state", {}).get("mean")
            if obs_state_mean is not None and (
                torch.isinf(obs_state_mean).any() or torch.isnan(obs_state_mean).any()
            ):
                print("⚠️  Found invalid normalization stats, replacing with dummy stats...")

                # Replace with dummy stats
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
                print("✓ Normalization layers updated with dummy stats")
    except Exception as e:
        print(f"⚠️  Error checking normalization stats, creating new ones: {e}")
        # Fallback: create new normalization layers
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
    batch = {
        "observation.state": torch.randn(
            batch_size, policy.config.state_dim, dtype=torch.float32, device=device
        ),
        "action": torch.randn(
            batch_size,
            policy.config.chunk_size,
            policy.config.action_dim,
            dtype=torch.float32,
            device=device,
        ),
        "task": ["Pick up the object"] * batch_size,
    }

    # Add images if they're in the config
    for key in policy.config.image_keys:
        batch[key] = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device)

    try:
        # Test forward pass
        policy.train()  # Set to training mode for forward pass with loss
        loss, loss_dict = policy.forward(batch)
        print("✓ Forward pass successful")
        print(f"  - Loss: {loss_dict['loss']:.4f}")
        print(f"  - Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "-" * 60)
    print("Testing inference with loaded model...")

    try:
        # Test action prediction
        policy.eval()  # Set to evaluation mode for inference
        with torch.no_grad():
            action = policy.select_action(batch)
        print("✓ Action prediction successful")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    except Exception as e:
        print(f"✗ Action prediction failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "=" * 60)
    print(f"✓ All tests passed for {model_name}!")
    print("=" * 60)
