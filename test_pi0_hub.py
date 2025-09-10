#!/usr/bin/env python

"""Test script to load PI0OpenPI model from HuggingFace hub and run inference."""

import torch

from lerobot.policies.pi0_openpi import PI0OpenPIPolicy


def test_hub_loading():
    """Test loading model from HuggingFace hub."""
    print("=" * 60)
    print("PI0OpenPI HuggingFace Hub Loading Test")
    print("=" * 60)

    # Model ID on HuggingFace hub
    model_id = "pepijn223/pi0_base_fp32"  # We made sure this config matches our code and `PI0OpenPIConfig` by uploading a model with push_pi0_to_hub.py and copying that config.

    print(f"\nLoading model from: {model_id}")
    print("-" * 60)

    try:
        # Load the model from HuggingFace hub with strict mode
        policy = PI0OpenPIPolicy.from_pretrained(
            model_id,
            strict=True,  # Ensure all weights are loaded correctly
        )
        print("✓ Model loaded successfully from HuggingFace hub")

        # Get model info
        print("\nModel configuration:")
        print(f"  - PaliGemma variant: {policy.config.paligemma_variant}")
        print(f"  - Action expert variant: {policy.config.action_expert_variant}")
        print(f"  - Action dimension: {policy.config.action_dim}")
        print(f"  - State dimension: {policy.config.state_dim}")
        print(f"  - Action horizon: {policy.config.action_horizon}")
        print(f"  - Device: {next(policy.parameters()).device}")
        print(f"  - Dtype: {next(policy.parameters()).dtype}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    print("\n" + "-" * 60)
    print("Testing forward pass with loaded model...")

    # Create dummy batch for testing
    batch_size = 1
    device = next(policy.parameters()).device

    # Create dummy dataset stats if not loaded with the model
    if not hasattr(policy, "normalize_inputs") or policy.normalize_inputs is None:
        from lerobot.policies.normalize import Normalize, Unnormalize

        dataset_stats = {
            "observation.state": {
                "mean": torch.zeros(policy.config.state_dim, device=device),
                "std": torch.ones(policy.config.state_dim, device=device),
            },
            "action": {
                "mean": torch.zeros(policy.config.action_dim, device=device),
                "std": torch.ones(policy.config.action_dim, device=device),
            },
        }
        policy.normalize_inputs = Normalize(
            policy.config.input_features, policy.config.normalization_mapping, dataset_stats
        )
        policy.normalize_targets = Normalize(
            policy.config.output_features, policy.config.normalization_mapping, dataset_stats
        )
        policy.unnormalize_outputs = Unnormalize(
            policy.config.output_features, policy.config.normalization_mapping, dataset_stats
        )

    # Create test batch
    batch = {
        "observation.state": torch.randn(
            batch_size, policy.config.state_dim, dtype=torch.float32, device=device
        ),
        "action": torch.randn(
            batch_size,
            policy.config.action_horizon,
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
        return False

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
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_hub_loading()
    exit(0 if success else 1)
