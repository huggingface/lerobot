"""Test script to verify PI0OpenPI policy integration with LeRobot vs the original implementation."""

import os

import torch

# NOTE: Assumes PYTHONPATH is set to include OpenPI src as per instructions.
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from lerobot.policies.pi0_openpi import PI0OpenPIConfig, PI0OpenPIPolicy

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 48  # Default for PI0 (non-pi05)
DEVICE = "cpu"  # Use CPU to avoid memory issues for testing

DUMMY_DATASET_STATS = {
    "observation.state": {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
    },
    "action": {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        "base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
        "left_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
        "right_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    },
}


class PI0BaseOriginalConfig:
    action_dim: int = DUMMY_ACTION_DIM
    action_horizon: int = DUMMY_ACTION_HORIZON
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "float32"
    pi05: bool = False
    dtype: str = "float32"


def instantiate_lerobot_pi0(from_pretrained: bool = False):
    if from_pretrained:
        # Load the policy first
        policy = PI0OpenPIPolicy.from_pretrained("pepijn223/pi0_base_fp32")
        # Then reinitialize the normalization with proper stats
        from lerobot.policies.normalize import Normalize, Unnormalize

        policy.normalize_inputs = Normalize(
            policy.config.input_features, policy.config.normalization_mapping, DUMMY_DATASET_STATS
        )
        policy.normalize_targets = Normalize(
            policy.config.output_features, policy.config.normalization_mapping, DUMMY_DATASET_STATS
        )
        policy.unnormalize_outputs = Unnormalize(
            policy.config.output_features, policy.config.normalization_mapping, DUMMY_DATASET_STATS
        )
    else:
        config = PI0OpenPIConfig(action_dim=DUMMY_ACTION_DIM, state_dim=DUMMY_STATE_DIM, dtype="float32")
        policy = PI0OpenPIPolicy(config, DUMMY_DATASET_STATS)
    policy.to(DEVICE)
    return policy


def instantiate_original_pi0(from_pretrained: bool = False, model_path: str = None):
    config = PI0BaseOriginalConfig()
    policy = PI0Pytorch(config)

    if from_pretrained:
        try:
            print("Loading converted PyTorch weights from HuggingFace Hub (pepijn223/pi0_base_fp32)...")

            # Download the model from HuggingFace Hub
            import safetensors.torch
            from huggingface_hub import snapshot_download

            # Download the entire repository
            if model_path and os.path.exists(model_path):
                cache_dir = model_path
                print(f"Using cached model from: {cache_dir}")
            else:
                cache_dir = snapshot_download(repo_id="pepijn223/pi0_base_fp32", repo_type="model")
                print(f"Downloaded model to: {cache_dir}")

            # Try to load safetensors format first
            model_file = os.path.join(cache_dir, "model.safetensors")
            if os.path.exists(model_file):
                state_dict = safetensors.torch.load_file(model_file)
                print(f"Loaded {len(state_dict)} parameters from safetensors")
            else:
                raise FileNotFoundError(f"No safetensors file found in {cache_dir}")

            # Load the state dict into the model
            missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"    - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
                    print(f"    ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"    - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"    - {key}")
                    print(f"    ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All pretrained weights loaded successfully!")
            else:
                print("Pretrained weights loaded with some missing/unexpected keys (this may be normal)")

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("   Using randomly initialized weights...")
            import traceback

            traceback.print_exc()

    policy.to(DEVICE)
    return policy


def create_dummy_data():
    batch_size = 2  # Reduce batch size for testing
    device = DEVICE

    # Use the exact same prompt for both implementations
    prompt = "Pick up the red block and place it in the bin"

    batch = {
        "observation.state": torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "action": torch.randn(
            batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, dtype=torch.float32, device=device
        ),
        # Create images in [-1, 1] range as expected by both implementations
        "observation.images.base_0_rgb": torch.randn(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ).clamp(-1, 1),
        "observation.images.left_wrist_0_rgb": torch.randn(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ).clamp(-1, 1),
        "observation.images.right_wrist_0_rgb": torch.randn(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ).clamp(-1, 1),
        # Add the task prompt for LeRobot - provide as list with single element to trigger expansion
        "task": [prompt],
    }
    return batch


def extract_lerobot_processed_inputs(lerobot_pi0, batch):
    """Extract the exact same processed inputs that LeRobot uses internally."""
    # Get the tokenized language from LeRobot's internal method
    lang_tokens, lang_masks = lerobot_pi0._tokenize_language(batch)

    # Get the preprocessed images from LeRobot's internal method
    images, img_masks = lerobot_pi0._preprocess_images(batch)

    # Create dummy token_ar_mask and token_loss_mask for original implementation
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    return images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask


class PI0Observation:
    """Observation class that matches the original OpenPI format."""

    def __init__(
        self,
        state,
        images,
        image_masks,
        tokenized_prompt,
        tokenized_prompt_mask,
        token_ar_mask,
        token_loss_mask,
    ):
        self.state = state
        self.images = images
        self.image_masks = image_masks
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


def create_original_observation_from_lerobot(lerobot_pi0, batch):
    """Create observation object compatible with original OpenPI using the exact same inputs as LeRobot."""
    _batch_size = batch["observation.state"].shape[0]
    _device = batch["observation.state"].device

    # Extract the exact same processed inputs that LeRobot uses
    images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask = (
        extract_lerobot_processed_inputs(lerobot_pi0, batch)
    )

    # Convert images list to dict with original OpenPI keys
    image_dict = {
        "base_0_rgb": images[0],
        "left_wrist_0_rgb": images[1],
        "right_wrist_0_rgb": images[2],
    }

    # Convert image masks list to dict with original OpenPI keys
    image_masks_dict = {
        "base_0_rgb": img_masks[0],
        "left_wrist_0_rgb": img_masks[1],
        "right_wrist_0_rgb": img_masks[2],
    }

    return PI0Observation(
        state=batch["observation.state"],
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )


def main():
    print("Initializing models...")
    lerobot_pi0 = instantiate_lerobot_pi0(from_pretrained=True)  # Load pretrained LeRobot model
    original_pi0 = instantiate_original_pi0(
        from_pretrained=True
    )  # Load pretrained OpenPI model from HuggingFace Hub

    print("Creating dummy data...")
    batch = create_dummy_data()

    print("Creating observation for original PI0 using LeRobot's exact preprocessing...")
    pi0_obs = create_original_observation_from_lerobot(lerobot_pi0, batch)

    # Verify both implementations get the same inputs
    print(f"Task prompt: '{batch['task'][0]}'")
    print(f"Tokenized prompt shape: {pi0_obs.tokenized_prompt.shape}")
    print(f"Image shapes: {[img.shape for img in pi0_obs.images.values()]}")
    print(f"State shape: {pi0_obs.state.shape}")

    print("Testing original PI0...")

    # Test training forward pass (returns loss)
    print("1. Training forward pass (computing loss):")
    original_pi0.train()
    original_loss = original_pi0(observation=pi0_obs, actions=batch["action"])
    print(f"   Loss shape: {original_loss.shape}, Mean loss: {original_loss.mean().item():.6f}")

    # Test inference (action sampling) with fixed noise for reproducibility
    print("2. Inference (action sampling):")
    original_pi0.eval()

    # Create the same noise for both implementations
    torch.manual_seed(42)  # Set seed for reproducibility
    batch_size = batch["observation.state"].shape[0]
    noise_shape = (batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM)
    fixed_noise = torch.randn(noise_shape, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        original_actions = original_pi0.sample_actions(
            device=DEVICE, observation=pi0_obs, noise=fixed_noise, num_steps=10
        )
    print(f"Original PI0 Actions shape: {original_actions.shape}")
    print(f"Original PI0 Actions mean: {original_actions.mean().item():.6f}")
    print(f"Original PI0 Actions std: {original_actions.std().item():.6f}")

    # Test LeRobot implementation with the same noise
    print("\nTesting LeRobot PI0...")
    lerobot_pi0.eval()

    # For LeRobot, we need to modify the batch to force the same noise
    # This is more complex since LeRobot generates noise internally
    torch.manual_seed(42)  # Set the same seed
    with torch.no_grad():
        # lerobot_pi0_actions = lerobot_pi0.select_action(batch)
        lerobot_pi0_actions = lerobot_pi0.predict_action_chunk(batch)
    print(f"LeRobot actions shape: {lerobot_pi0_actions.shape}")
    print(f"LeRobot actions mean: {lerobot_pi0_actions.mean().item():.6f}")
    print(f"LeRobot actions std: {lerobot_pi0_actions.std().item():.6f}")

    print("\nComparing implementations:")
    print(f"Original actions shape: {original_actions.shape}")
    print(f"LeRobot actions shape: {lerobot_pi0_actions.shape}")

    # Compare the first action step (since LeRobot select_action returns a single step)
    print(f"Actions close (atol=1e-4): {torch.allclose(lerobot_pi0_actions, original_actions, atol=1e-4)}")
    print(f"Actions close (atol=1e-2): {torch.allclose(lerobot_pi0_actions, original_actions, atol=1e-2)}")
    print(f"Max absolute difference: {torch.abs(lerobot_pi0_actions - original_actions).max().item():.6f}")

    print("\nOriginal PI0 test completed successfully!")


if __name__ == "__main__":
    main()
