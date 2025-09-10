import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers.model_debugging_utils import model_addition_debugger_context

from lerobot.configs.policies import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

RANDOM_SEED = 42  # Set to fixed value for reproducible results


def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    print(f"All random seeds set to {seed} for reproducible results (deterministic mode enabled)")


# Set seeds at the start
set_all_seeds(RANDOM_SEED)

config_model_path = "lerobot/pi0"  # Use config from official model
official_model_path = "lerobot/pi0"  # Official model
custom_model_path = "pepijn223/pi0_base_fp32"  # Custom model to compare # pepijn223/pi0_base_fp32
device = "mps"

USE_FULL_TENSORS = True
SAVE_TENSORS_TO_DISK = False

# Model transformation and upload settings
SAVE_TRANSFORMED_MODEL = True  # Set to True to save the transformed model
UPLOAD_TO_HUB = True  # Set to True to upload to HuggingFace Hub
TRANSFORMED_MODEL_NAME = "pepijn223/pi0_base_fp32_lerobot_format"  # Target repo name
COMMIT_MESSAGE = "Add transformed PI0 model with correct key format for lerobot"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
debug_path = os.path.join("debug_outputs", f"pi0_debug_direct_{timestamp}")
os.makedirs(debug_path, exist_ok=True)
print(f"Model debugging enabled - outputs will be saved to: {debug_path}")

# Download and load the config manually to avoid draccus parsing issues
config_file = hf_hub_download(repo_id=config_model_path, filename="config.json")
with open(config_file) as f:
    config_dict = json.load(f)

# Remove the 'type' field that causes draccus issues
if "type" in config_dict:
    config_dict.pop("type")
    print("Removed 'type' field from config")

# Create shared PI0Config
print("Creating shared PI0Config...")
shared_config = PI0Config(**config_dict)


def load_policy_with_weights(
    model_path: str, config: PI0Config, model_name: str, apply_transformations: bool = False
):
    """Load a policy with specified weights but shared config."""
    print(f"\n=== Loading {model_name} from {model_path} ===")

    # Set deterministic seed before creating the policy to ensure identical initialization
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    policy = PI0Policy(config)

    # Download and load weights
    model_file = hf_hub_download(repo_id=model_path, filename="model.safetensors")
    print(f"Downloaded {model_name} weights to: {model_file}")

    # Load state dict and apply transformations
    print(f"Investigating safetensors file: {model_file}")

    # First, check what's in the metadata
    try:
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            all_keys_in_file = f.keys()
            print(f"   Total keys in safetensors file: {len(list(all_keys_in_file))}")

            # Check for embed_tokens in the file keys
            embed_keys_in_file = [k for k in f.keys() if "embed_tokens" in k]
            print(f"   embed_tokens keys in safetensors: {embed_keys_in_file}")

            if metadata:
                print(f"   Metadata exists: {list(metadata.keys()) if metadata else 'None'}")
    except Exception as e:
        print(f"   Could not inspect safetensors file directly: {e}")

    # Now load normally and see what we get
    state_dict = load_file(model_file)
    print(f"   Keys loaded by load_file(): {len(state_dict)} keys")

    # Check for embed_tokens in loaded state_dict
    loaded_embed_keys = [k for k in state_dict.keys() if "embed_tokens" in k]
    print(f"   embed_tokens keys in loaded state_dict: {loaded_embed_keys}")

    # Check if we need to add "model." prefix (for custom models that don't have it)
    sample_key = next(iter(state_dict.keys()))
    if not sample_key.startswith("model."):
        print(f"Adding 'model.' prefix to all keys (detected format: {sample_key})")
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    # IMPORTANT: Call PI0Policy._transform_state_dict_keys AFTER adding model. prefix
    # This ensures tied weights logic can find the correct key pattern
    transformed_state_dict = PI0Policy._transform_state_dict_keys(state_dict)

    # Apply specific PaliGemma key transformations only for custom models
    if apply_transformations:
        print("Applying custom model key transformations...")

        # First, let's debug what keys we actually have
        all_keys = list(transformed_state_dict.keys())
        sample_keys = all_keys[:10]
        print(f"Sample keys to transform: {sample_keys}")

        # Look for specific keys we need to transform and missing keys
        embed_tokens_keys = [k for k in all_keys if "embed_tokens" in k]
        embedding_keys = [k for k in all_keys if "embed" in k]
        lm_head_keys = [k for k in all_keys if "lm_head" in k]
        paligemma_keys = [
            k for k in all_keys if "paligemma_with_expert.paligemma" in k and "gemma_expert" not in k
        ]
        language_model_keys = [k for k in all_keys if "language_model" in k]

        print(f"Found embed_tokens keys: {embed_tokens_keys}")
        print(f"Found any embedding keys: {embedding_keys}")
        print(f"Found lm_head keys: {lm_head_keys}")
        print(
            f"Found paligemma keys (non-expert): {paligemma_keys[:5]}{'...' if len(paligemma_keys) > 5 else ''}"
        )
        print(
            f"Found language_model keys: {language_model_keys[:5]}{'...' if len(language_model_keys) > 5 else ''}"
        )
        print(f"Total keys in model: {len(all_keys)}")

        # Check if the embed_tokens is in gemma_expert instead
        gemma_expert_embed = [k for k in all_keys if "gemma_expert" in k and "embed_tokens" in k]
        print(f"Found gemma_expert embed_tokens keys: {gemma_expert_embed}")

        # Check what we're missing and what we actually have
        expected_embed_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        if expected_embed_key not in all_keys:
            print(f" Missing expected embed_tokens key: {expected_embed_key}")

            # Let's see what keys we actually have for debugging
            print("Debugging: Looking for any embedding-related keys...")
            all_embed_related = [k for k in all_keys if "embed" in k.lower()]
            print(f"Keys containing 'embed': {all_embed_related}")

            # Look for any keys that might contain embeddings
            potential_embed_keys = [
                k for k in all_keys if any(word in k for word in ["embed", "embedding", "token"])
            ]
            print(f"   Potential embedding keys: {potential_embed_keys}")

            # Try to find a suitable replacement
            if gemma_expert_embed:
                print(f"   Will try to copy from: {gemma_expert_embed[0]}")
            else:
                print("   No gemma_expert embed_tokens found either!")

                # Check if there's an embed_tokens in the gemma_expert that we missed
                gemma_keys = [k for k in all_keys if "gemma_expert" in k]
                print(f"   First 10 gemma_expert keys: {gemma_keys[:10]}")

                # Check if there are any token-related keys in gemma_expert
                token_keys = [k for k in all_keys if "gemma_expert" in k and "token" in k.lower()]
                print(f"   Gemma expert token-related keys: {token_keys}")

                # Check for any keys that look like they might be embeddings
                possible_embeds = [
                    k
                    for k in all_keys
                    if any(
                        pattern in k.lower() for pattern in ["embed_token", "embedding", "wte", "word_embed"]
                    )
                ]
                print(f"   Possible embedding alternatives: {possible_embeds}")

        final_state_dict = {}
        transformation_count = 0

        for key, value in transformed_state_dict.items():
            new_key = key
            original_key = key

            # Transform vision tower keys: ADD .model between paligemma and vision_tower
            if "paligemma_with_expert.paligemma.vision_tower.vision_model" in new_key:
                new_key = new_key.replace(
                    "paligemma_with_expert.paligemma.vision_tower.vision_model",
                    "paligemma_with_expert.paligemma.model.vision_tower.vision_model",
                )
                print(f"Transformed vision key: {original_key} -> {new_key}")
                transformation_count += 1

            # Transform multi_modal_projector keys: ADD .model between paligemma and multi_modal_projector
            elif "paligemma_with_expert.paligemma.multi_modal_projector" in new_key:
                new_key = new_key.replace(
                    "paligemma_with_expert.paligemma.multi_modal_projector",
                    "paligemma_with_expert.paligemma.model.multi_modal_projector",
                )
                print(f"Transformed multi_modal_projector key: {original_key} -> {new_key}")
                transformation_count += 1

            # NO transformation needed for language_model keys - they're already correct!
            # The custom model already has: paligemma.model.language_model.* which is what we need

            # NO transformation needed for lm_head - it should stay as paligemma.lm_head

            final_state_dict[new_key] = value

        print(f"Applied {transformation_count} key transformations")
        transformed_state_dict = final_state_dict
    else:
        print("No transformations applied (official model format)")

    # Debug: show what keys the policy expects vs what we have
    policy_keys = set(policy.state_dict().keys())
    provided_keys = set(transformed_state_dict.keys())

    missing_in_provided = policy_keys - provided_keys
    extra_in_provided = provided_keys - policy_keys

    print(f"Policy expects {len(policy_keys)} keys, we provide {len(provided_keys)} keys")
    if missing_in_provided:
        print(
            f"   Missing from provided: {list(missing_in_provided)[:5]}{'...' if len(missing_in_provided) > 5 else ''}"
        )
    if extra_in_provided:
        print(
            f"   Extra in provided: {list(extra_in_provided)[:5]}{'...' if len(extra_in_provided) > 5 else ''}"
        )

    # Load the weights into the policy
    msg = policy.load_state_dict(transformed_state_dict, strict=True)
    print(
        f"{model_name} - Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}"
    )

    if msg.missing_keys:
        print(
            f"   Actually missing keys: {list(msg.missing_keys)[:3]}{'...' if len(msg.missing_keys) > 3 else ''}"
        )
    if msg.unexpected_keys:
        print(
            f"   Actually unexpected keys: {list(msg.unexpected_keys)[:3]}{'...' if len(msg.unexpected_keys) > 3 else ''}"
        )

    # Set deterministic mode and move to device
    policy = policy.to(device)
    policy.eval()

    # Reset the policy to ensure identical internal state
    policy.reset()

    return policy


# Load both models with shared config
print("Loading both models with shared config...")
official_policy = load_policy_with_weights(
    official_model_path, shared_config, "Official Model", apply_transformations=False
)
custom_policy = load_policy_with_weights(
    custom_model_path, shared_config, "Custom Model", apply_transformations=True
)

print("\nBoth models loaded successfully!")
print(f"Shared config: {shared_config}")
print(f"Device: {device}")


# Configure input features for both policies since they're not set by default in pretrained models
def configure_policy_features(policy: PI0Policy):
    """Configure input and output features for a policy."""
    policy.config.input_features[OBS_IMAGE] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),  # Channel-first RGB image
    )

    policy.config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(8,),  # 8-dimensional state vector
    )

    policy.config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(8,),  # 8-dimensional action vector
    )

    # Add dummy normalization buffers to the policy (like openpi does with norm_stats)
    if hasattr(policy, "normalize_inputs"):
        # For observation.state (8-dim state vector)
        policy.normalize_inputs.register_buffer(
            f"buffer_{OBS_STATE.replace('.', '_')}_mean", torch.zeros(8, device=device)
        )
        policy.normalize_inputs.register_buffer(
            f"buffer_{OBS_STATE.replace('.', '_')}_std", torch.ones(8, device=device)
        )

        # For observation.image (3x224x224 image)
        policy.normalize_inputs.register_buffer(
            f"buffer_{OBS_IMAGE.replace('.', '_')}_mean", torch.zeros(3, 224, 224, device=device)
        )
        policy.normalize_inputs.register_buffer(
            f"buffer_{OBS_IMAGE.replace('.', '_')}_std", torch.ones(3, 224, 224, device=device)
        )


print("Configuring features for both policies...")
configure_policy_features(official_policy)
configure_policy_features(custom_policy)

# Verify that the models have identical parameters
print("\n=== Model Parameter Comparison ===")
official_params = dict(official_policy.named_parameters())
custom_params = dict(custom_policy.named_parameters())

param_differences = []
for name in official_params.keys():
    if name not in custom_params:
        param_differences.append(f"Missing parameter in custom model: {name}")
    else:
        diff = torch.abs(official_params[name] - custom_params[name]).max().item()
        if diff > 1e-8:
            param_differences.append(f"Parameter {name}: max difference = {diff:.2e}")

for name in custom_params.keys():
    if name not in official_params:
        param_differences.append(f"Extra parameter in custom model: {name}")

if param_differences:
    print("Parameter differences found:")
    for diff in param_differences[:10]:  # Show first 10 differences
        print(f"   {diff}")
    if len(param_differences) > 10:
        print(f"   ... and {len(param_differences) - 10} more differences")
else:
    print("All model parameters are identical!")


# Get the raw models for direct comparison
official_raw_model = official_policy.model
custom_raw_model = custom_policy.model
print("\n=== Model Details ===")
print(f"Official raw model type: {type(official_raw_model)}")
print(f"Custom raw model type: {type(custom_raw_model)}")
print(f"Official model device: {next(official_raw_model.parameters()).device}")
print(f"Custom model device: {next(custom_raw_model.parameters()).device}")

# Create lerobot-format input data (similar to DROID format from openpi example)
example = {
    "joint_position": np.zeros(7, dtype=np.float32),
    "gripper_position": np.array([0.0], dtype=np.float32),
    "image": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
    "task": "pick up the object",
}

print(f"\nProvided input keys: {list(example.keys())}")

print("\nPreparing inputs for direct model call...")

# Apply input transformation (similar to openpi's policy._input_transform)
transformed_example = {}
# Combine joint and gripper positions into state
transformed_example[OBS_STATE] = np.concatenate([example["joint_position"], example["gripper_position"]])
transformed_example[OBS_IMAGE] = example["image"]
transformed_example["task"] = example["task"]

# Convert to PyTorch tensors and add batch dimension (as openpi example does)
# Device is already defined above, use the official model device for consistency
pytorch_inputs = {}
for key, value in transformed_example.items():
    if isinstance(value, np.ndarray):
        tensor_value = torch.from_numpy(value).to(device)
        # Add batch dimension
        if tensor_value.dim() > 0:
            tensor_value = tensor_value.unsqueeze(0)
        pytorch_inputs[key] = tensor_value
    elif isinstance(value, str):
        pytorch_inputs[key] = [value]  # Convert to list format expected by policy
    else:
        pytorch_inputs[key] = value

# Convert image from HWC to CHW format for lerobot
if OBS_IMAGE in pytorch_inputs:
    img = pytorch_inputs[OBS_IMAGE]
    if img.dim() == 4 and img.shape[-1] == 3:  # BHWC -> BCHW
        img = img.permute(0, 3, 1, 2)
    # Convert to float and normalize to [0, 1] range
    img = img.float() / 255.0
    pytorch_inputs[OBS_IMAGE] = img

print(f"Transformed input keys: {list(pytorch_inputs.keys())}")
for key, value in pytorch_inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape} {value.dtype}")
    else:
        print(f"  {key}: {type(value)} - {value}")

# Reset both policies (clears the action queue)
official_policy.reset()
custom_policy.reset()

# Prepare inputs using the official policy (both models should have same preprocessing)
print("Preparing inputs for both models...")
images, img_masks = official_policy.prepare_images(pytorch_inputs)
lang_tokens, lang_masks = official_policy.prepare_language(pytorch_inputs)
state = official_policy.prepare_state(pytorch_inputs)

print("Prepared inputs:")
print(f"  Images: {len(images)} images")
print(f"  Language tokens shape: {lang_tokens.shape}")
print(f"  State shape: {state.shape}")
for i, img in enumerate(images):
    print(f"  Image {i} shape: {img.shape}")
for i, mask in enumerate(img_masks):
    print(f"  Image mask {i} shape: {mask.shape}")

# Compare both models with identical inputs
print("\n🚀 Running MODEL COMPARISON...")

# Force torch.no_grad for consistent comparison
with torch.no_grad():
    # Ensure reproducible noise generation for both models
    torch.manual_seed(RANDOM_SEED)

    # Generate synthetic noise and time for the forward call
    batch_size = 1
    actions_shape = (
        batch_size,
        official_raw_model.config.n_action_steps,
        official_raw_model.config.max_action_dim,
    )

    # Generate noise and time using direct PyTorch operations instead of model methods
    # This avoids any potential model-specific randomness
    torch.manual_seed(RANDOM_SEED)
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=actions_shape,
        dtype=torch.float32,
        device=device,
    )

    # Generate time using the same distribution as PI0FlowMatching.sample_time
    torch.manual_seed(RANDOM_SEED)  # Reset for consistent time
    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    time_beta = beta_dist.sample((batch_size,)).to(device=device, dtype=torch.float32)
    time = time_beta * 0.999 + 0.001

    print("\n=== Generated Inputs ===")
    print(f"   Action shape: {actions_shape}")
    print(f"   Noise shape: {noise.shape}")
    print(f"   Time value: {time.item():.6f}")
    print(f"   Noise sample (first 5 values): {noise.flatten()[:5].tolist()}")

    # Create dummy actions for forward pass (required for training forward)
    dummy_actions = torch.zeros(actions_shape, dtype=torch.float32, device=device)

    print("\n=== Running Forward Passes ===")

    print("Running with model_addition_debugger_context for detailed analysis...")
    # Create separate debug paths for each model
    official_debug_path = os.path.join(debug_path, "official_model")
    custom_debug_path = os.path.join(debug_path, "custom_model")
    os.makedirs(official_debug_path, exist_ok=True)
    os.makedirs(custom_debug_path, exist_ok=True)
    # Set deterministic mode for forward pass
    torch.manual_seed(RANDOM_SEED)
    # Run official model with debugger
    print("Running Official Model forward pass with debugger...")
    with model_addition_debugger_context(
        official_raw_model,
        debug_path=official_debug_path,
        do_prune_layers=False,  # Output ALL layers
        use_repr=not SAVE_TENSORS_TO_DISK,
    ):
        official_loss = official_raw_model.forward(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=dummy_actions,
            noise=noise,
            time=time,
        )
    # Reset seed before second forward pass to ensure any internal randomness is identical
    torch.manual_seed(RANDOM_SEED)
    # Run custom model with debugger
    print("Running Custom Model forward pass with debugger...")
    with model_addition_debugger_context(
        custom_raw_model,
        debug_path=custom_debug_path,
        do_prune_layers=False,  # Output ALL layers
        use_repr=not SAVE_TENSORS_TO_DISK,
    ):
        custom_loss = custom_raw_model.forward(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=dummy_actions,
            noise=noise,
            time=time,
        )

    print(f"Official model debug outputs saved to: {official_debug_path}")
    print(f"Custom model debug outputs saved to: {custom_debug_path}")

    print("\n=== Output Comparison ===")
    print(f"Official model loss shape: {official_loss.shape}")
    print(f"Custom model loss shape: {custom_loss.shape}")

    # Compare outputs
    loss_diff = torch.abs(official_loss - custom_loss)

    print("\n=== Detailed Comparison ===")
    print("Loss difference stats:")
    print(f"  Mean absolute difference: {loss_diff.mean().item():.8f}")
    print(f"  Max absolute difference: {loss_diff.max().item():.8f}")
    print(f"  Min absolute difference: {loss_diff.min().item():.8f}")
    print(f"  Standard deviation of difference: {loss_diff.std().item():.8f}")

    # Show some actual values for comparison
    print("\nSample output values:")
    print(f"  Official model (first 5): {official_loss.flatten()[:5].tolist()}")
    print(f"  Custom model (first 5): {custom_loss.flatten()[:5].tolist()}")
    print(f"  Difference (first 5): {loss_diff.flatten()[:5].tolist()}")

    # Determine if models are equivalent
    are_equivalent = loss_diff.max().item() < 1e-6
    print(f"\nModels are {'EQUIVALENT' if are_equivalent else 'DIFFERENT'}")
    print(f"   (Max difference: {loss_diff.max().item():.8f}, Threshold: 1e-6)")

    print(f"\nDetailed debugging outputs saved to: {debug_path}")
    # Save comparison results
    comparison_results = {
        "official_loss_stats": {
            "shape": list(official_loss.shape),
            "mean": official_loss.mean().item(),
            "std": official_loss.std().item(),
            "min": official_loss.min().item(),
            "max": official_loss.max().item(),
        },
        "custom_loss_stats": {
            "shape": list(custom_loss.shape),
            "mean": custom_loss.mean().item(),
            "std": custom_loss.std().item(),
            "min": custom_loss.min().item(),
            "max": custom_loss.max().item(),
        },
        "difference_stats": {
            "mean_abs_diff": loss_diff.mean().item(),
            "max_abs_diff": loss_diff.max().item(),
            "min_abs_diff": loss_diff.min().item(),
            "std_diff": loss_diff.std().item(),
            "are_equivalent": are_equivalent,
        },
    }

    comparison_file = os.path.join(debug_path, "model_comparison_results.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"   Comparison results saved to: {comparison_file}")

# Save and upload transformed model if requested
if SAVE_TRANSFORMED_MODEL:
    print("\nSaving Transformed Model...")
    if are_equivalent:
        print("Models are equivalent - proceeding with transformation and upload")
    else:
        print("Models are NOT equivalent, but proceeding with upload anyway")
        print(f"   Max difference: {loss_diff.max().item():.2e}")
        print("   This might be useful for debugging or partial transformations")

    # Create timestamp for README
    transformation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Use the already working custom policy as the base for transformation
        print("Using already working custom policy as base for transformed model...")

        # Deep copy the custom policy to create the transformed version
        from copy import deepcopy

        transformed_policy = deepcopy(custom_policy)

        print("Custom policy copied successfully - no additional configuration needed")

        # Save locally first
        local_save_path = "./transformed_pi0_model"
        print(f"Saving transformed model locally to: {local_save_path}")
        transformed_policy.save_pretrained(local_save_path, safe_serialization=True)

        # Save the tokenizer as well (required for complete model)
        transformed_policy.language_tokenizer.save_pretrained(local_save_path)

        # Create a README with transformation details
        readme_content = f"""
# PI0 Model - LeRobot Compatible Format

This model is a transformed version of `{custom_model_path}` with key names corrected to match the official LeRobot PI0 format.

## Transformation Applied

The original model had a different key naming convention. This model applies the following transformations:

1. **Model prefix**: Added `model.` prefix to all parameter keys
2. **Tied weights**: Applied PI0Policy's built-in tied weights logic to create `embed_tokens.weight` from `lm_head.weight`
3. **Key structure**: Applied standard PI0 key transformations for compatibility

## Verification

{"This transformed model produces **identical outputs**" if are_equivalent else "This transformed model has **slightly different outputs**"} (max difference = {loss_diff.max().item():.2e}) compared to the official model `{official_model_path}` when tested with the same inputs.
{"**Models are EQUIVALENT** (difference < 1e-6)" if are_equivalent else "**Models are NOT equivalent** (difference >= 1e-6) - use with caution"}

## Usage

```python
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# Load the model
policy = PI0Policy.from_pretrained("{TRANSFORMED_MODEL_NAME}")

# Use for inference
action = policy.select_action(observation_batch)
```

## Original Model

- **Source**: {custom_model_path}
- **Verified Against**: {official_model_path}

## Technical Details

- **Total Parameters**: {sum(p.numel() for p in transformed_policy.parameters()):,}
- **Model Type**: PI0FlowMatching with PaliGemma + Expert Gemma
- **Configuration**: Matches official PI0 configuration
"""

        readme_path = os.path.join(local_save_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content.strip())

        print(f"Model saved locally to: {local_save_path}")

        # Upload to HuggingFace Hub if requested
        if UPLOAD_TO_HUB:
            print(f"\nUploading to HuggingFace Hub: {TRANSFORMED_MODEL_NAME}")

            try:
                # Push to hub
                transformed_policy.push_to_hub(
                    repo_id=TRANSFORMED_MODEL_NAME,
                    commit_message=COMMIT_MESSAGE,
                    private=False,  # Make it public
                    safe_serialization=True,
                )

                print(f"Model successfully uploaded to: https://huggingface.co/{TRANSFORMED_MODEL_NAME}")
                print("You can now use this model directly without any transformations!")
                print("\n Usage:")
                print("   from lerobot.policies.pi0.modeling_pi0 import PI0Policy")
                print(f"   policy = PI0Policy.from_pretrained('{TRANSFORMED_MODEL_NAME}')")

            except Exception as upload_error:
                print(f"Failed to upload to HuggingFace Hub: {upload_error}")
                print(f"You can manually upload the model from: {local_save_path}")
                print("   Or set UPLOAD_TO_HUB = False and upload later")

    except Exception as e:
        import traceback

        print(f"Error saving transformed model: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        print("The model transformation logic works, but saving failed")

else:
    print("\nModel transformation and upload disabled (SAVE_TRANSFORMED_MODEL = False)")
