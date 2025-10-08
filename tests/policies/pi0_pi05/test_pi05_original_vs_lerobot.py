#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script to verify PI0OpenPI policy integration with LeRobot vs the original implementation, only meant to be run locally!"""

import os
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

# Skip if openpi or transformers is not available
pytest.importorskip("openpi")
pytest.importorskip("transformers")

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local OpenPI installation and is not meant for CI",
)

from openpi.models_pytorch import preprocessing_pytorch as openpi_preprocessing  # noqa: E402

# NOTE: Assumes PYTHONPATH is set to include OpenPI src as per instructions.
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402

# TODO: ADDING DEFAULT IMAGES_FEATURES TO CONFIG
DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200
DEVICE = "cpu"  # Use CPU to avoid memory issues for testing

DUMMY_DATASET_STATS = {
    "observation.state": {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
        "q01": torch.zeros(DUMMY_STATE_DIM),
        "q99": torch.ones(DUMMY_STATE_DIM),
    },
    "action": {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
        "q01": torch.zeros(DUMMY_ACTION_DIM),
        "q99": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        "base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "left_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "right_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    },
}


class PI05BaseOriginalConfig:
    action_dim: int = DUMMY_ACTION_DIM
    action_horizon: int = DUMMY_ACTION_HORIZON
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "float32"
    pi05: bool = True
    dtype: str = "float32"


def instantiate_lerobot_pi05(
    from_pretrained: bool = False,
) -> tuple[
    PI05Policy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if from_pretrained:
        # Load the policy first
        policy = PI05Policy.from_pretrained(pretrained_name_or_path="lerobot/pi05_base", strict=True)
    else:
        config = PI05Config(max_action_dim=DUMMY_ACTION_DIM, max_state_dim=DUMMY_STATE_DIM, dtype="float32")
        policy = PI05Policy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config, dataset_stats=DUMMY_DATASET_STATS
    )
    return (policy, preprocessor, postprocessor)


def instantiate_original_pi05(from_pretrained: bool = False, model_path: str | None = None):
    config = PI05BaseOriginalConfig()
    policy = PI0Pytorch(config)

    if from_pretrained:
        try:
            print("Loading converted PyTorch weights from HuggingFace Hub (lerobot/pi05_base)...")

            # Download the model from HuggingFace Hub
            import safetensors.torch
            from huggingface_hub import snapshot_download

            # Download the entire repository
            if model_path and os.path.exists(model_path):
                cache_dir = model_path
                print(f"Using cached model from: {cache_dir}")
            else:
                cache_dir = snapshot_download(repo_id="lerobot/pi05_base", repo_type="model")
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
        # Create images in [0, 1] range as expected by LeRobot (will be converted to [-1, 1] internally)
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "observation.images.left_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "observation.images.right_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        # Add the task prompt for LeRobot - provide as list with single element to trigger expansion
        "task": [prompt for _ in range(batch_size)],
    }
    return batch


def extract_lerobot_processed_inputs(lerobot_pi0, batch):
    """Extract the exact same processed inputs that LeRobot uses internally."""
    # Get the tokenized language from LeRobot's internal method
    lang_tokens, lang_masks = lerobot_pi0._tokenize_language(batch)

    # Get the preprocessed images from LeRobot's internal method
    images, img_masks = lerobot_pi0._preprocess_images(batch, train=False)

    # Create dummy token_ar_mask and token_loss_mask for original implementation
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    return images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask


class PI05Observation:
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


def create_original_observation_with_openpi_preprocessing(batch):
    """Create observation object for OpenPI using OpenPI's own preprocessing with pi05 state tokenizer."""
    batch_size = batch["observation.state"].shape[0]
    device = batch["observation.state"].device

    # Create tokenizer for OpenPI (same as LeRobot uses)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # Get task description (pi05 processor handles all text formatting)
    tasks = batch.get("task", ["Pick up the object"] * batch_size)
    if isinstance(tasks, str):
        tasks = [tasks] * batch_size
    elif len(tasks) == 1:
        tasks = tasks * batch_size

    # Use pi05 state and input tokenizer logic (same as Pi05PrepareStateTokenizerProcessorStep)
    state = batch["observation.state"]
    state = deepcopy(state)

    # Prepare state (pad to max_state_dim)
    from lerobot.policies.pi05.modeling_pi05 import pad_vector

    state = pad_vector(state, DUMMY_STATE_DIM)

    # Normalize state to [-1, 1] range if needed (assuming it's already normalized from normalize_inputs)
    # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
    state_np = state.cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    # Create pi05-formatted prompts that include state information
    full_prompts = []
    for i, task in enumerate(tasks):
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized_states[i]))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        full_prompts.append(full_prompt)

    # Tokenize with max_length padding to match OpenPI's expected format
    tokenized = tokenizer(
        full_prompts,
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=DUMMY_MAX_TOKEN_LEN,
        return_tensors="pt",
    )

    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].to(device, dtype=torch.bool)

    # Create dummy token_ar_mask and token_loss_mask for OpenPI
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    # Convert LeRobot images format to OpenPI format (convert [0,1] to [-1,1] range)
    image_dict = {
        "base_0_rgb": batch["observation.images.base_0_rgb"] * 2.0 - 1.0,
        "left_wrist_0_rgb": batch["observation.images.left_wrist_0_rgb"] * 2.0 - 1.0,
        "right_wrist_0_rgb": batch["observation.images.right_wrist_0_rgb"] * 2.0 - 1.0,
    }

    # Create image masks (all ones for real images)
    image_masks_dict = {}
    for key in image_dict:
        image_masks_dict[key] = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Create raw observation object (before preprocessing)
    raw_observation = PI05Observation(
        state=batch["observation.state"],
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )

    # Now use OpenPI's preprocessing
    processed_obs = openpi_preprocessing.preprocess_observation_pytorch(raw_observation, train=False)

    return processed_obs


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

    return PI05Observation(
        state=batch["observation.state"],
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )


def test_pi05_original_vs_lerobot():
    """Test PI05 original implementation vs LeRobot implementation."""
    print("Initializing models...")
    lerobot_pi05, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_pi05(
        from_pretrained=True
    )  # Load pretrained LeRobot model
    original_pi0 = instantiate_original_pi05(
        from_pretrained=True
    )  # Load pretrained OpenPI model from HuggingFace Hub

    print("Creating dummy data...")
    batch = create_dummy_data()
    batch_lerobot = deepcopy(batch)

    # Test each model with its own preprocessing (more realistic end-to-end test)
    print("\nTest each model with its own preprocessing")
    print("Creating observation for OpenPI using OpenPI's own preprocessing...")
    pi0_obs_openpi = create_original_observation_with_openpi_preprocessing(batch)

    print(f"Task prompt: '{batch['task'][0]}'")
    print(f"OpenPI tokenized prompt shape: {pi0_obs_openpi.tokenized_prompt.shape}")
    print(f"OpenPI image shapes: {[img.shape for img in pi0_obs_openpi.images.values()]}")
    print(f"OpenPI state shape: {pi0_obs_openpi.state.shape}")

    print("Testing OpenPI with own preprocessing...")
    original_pi0.eval()
    torch.manual_seed(42)  # Set seed for reproducibility
    batch_size = batch["observation.state"].shape[0]
    noise_shape = (batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM)
    fixed_noise = torch.randn(noise_shape, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        openpi_actions = original_pi0.sample_actions(
            device=DEVICE, observation=pi0_obs_openpi, noise=fixed_noise, num_steps=10
        )
        openpi_actions_unit = openpi_actions[:, 0, :]
    print(f"OpenPI (own preprocessing) Actions shape: {openpi_actions.shape}")
    print(f"OpenPI (own preprocessing) Actions unit shape: {openpi_actions_unit.shape}")
    print(f"OpenPI (own preprocessing) Actions mean: {openpi_actions.mean().item():.6f}")
    print(f"OpenPI (own preprocessing) Actions std: {openpi_actions.std().item():.6f}")

    print("Testing LeRobot with own preprocessing...")
    lerobot_pi05.eval()
    torch.manual_seed(42)  # Set the same seed

    batch_lerobot_processed = lerobot_preprocessor(batch_lerobot)
    with torch.no_grad():
        lerobot_actions_own = lerobot_pi05.predict_action_chunk(
            batch_lerobot_processed
        )  # batch_size, n_action_steps, action_dim
        lerobot_actions_unit = lerobot_actions_own[:, 0, :]
    print(f"LeRobot (own preprocessing) Actions shape: {lerobot_actions_own.shape}")
    print(f"LeRobot (own preprocessing) Actions unit shape: {lerobot_actions_unit.shape}")
    print(f"LeRobot (own preprocessing) Actions mean: {lerobot_actions_own.mean().item():.6f}")
    print(f"LeRobot (own preprocessing) Actions std: {lerobot_actions_own.std().item():.6f}")

    print("\nComparing end-to-end implementations:")
    print(f"Actions close (atol=1e-4): {torch.allclose(lerobot_actions_own, openpi_actions, atol=1e-4)}")
    print(f"Actions close (atol=1e-2): {torch.allclose(lerobot_actions_own, openpi_actions, atol=1e-2)}")
    print(f"Max absolute difference: {torch.abs(lerobot_actions_own - openpi_actions).max().item():.6f}")

    assert torch.allclose(lerobot_actions_own, openpi_actions, atol=1e-4)
    assert torch.allclose(lerobot_actions_own, openpi_actions, atol=1e-2)
    assert torch.abs(lerobot_actions_own - openpi_actions).max().item() < 1e-4
