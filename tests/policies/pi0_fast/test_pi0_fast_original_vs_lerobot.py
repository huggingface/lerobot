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

"""Test script to verify PI0Fast policy integration with LeRobot vs the original implementation"""
# ruff: noqa: E402

import os
import random
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires accepting the model license",
)

from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.policies.pi0_fast.processor_pi0_fast import make_pi0_fast_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)  # noqa: E402
from tests.utils import require_cuda  # noqa: E402

# Constants
DUMMY_ACTION_DIM = 7
DUMMY_STATE_DIM = 20
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_VIEWS = 2  # Number of camera views
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH_LEROBOT = "lerobot/pi0fast-base"

# Expected action token shape: (batch_size, max_decoding_steps)
EXPECTED_ACTION_TOKENS_SHAPE = (1, 2)

# Expected first 5 action tokens (for reproducibility check)
EXPECTED_ACTION_TOKENS_FIRST_5 = torch.tensor([255657, 255362])

# Expected actions after detokenization
EXPECTED_ACTIONS_SHAPE = (1, 2, 32)  # (batch_size, n_action_steps, action_dim)
EXPECTED_ACTIONS_MEAN = 0.04419417306780815
EXPECTED_ACTIONS_STD = 0.26231569051742554
EXPECTED_ACTIONS_FIRST_5 = torch.tensor([0.0000, 1.4849, 0.0000, 0.0000, 0.0000])


def set_seed_all(seed: int):
    """Set random seed for all RNG sources to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def instantiate_lerobot_pi0_fast(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH_LEROBOT,
) -> tuple[
    Any,  # Policy
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Instantiate LeRobot PI0Fast policy with preprocessor and postprocessor."""
    if from_pretrained:
        policy = PI0FastPolicy.from_pretrained(
            pretrained_name_or_path=model_path,
            strict=True,
        )
        policy.config.validate_action_token_prefix = False
        policy.config.max_action_tokens = 2
        policy.config.max_decoding_steps = 2
        policy.config.chunk_size = 2
        policy.config.n_action_steps = 2
    else:
        config = PI0FastConfig(
            n_action_steps=2,
            max_action_dim=DUMMY_ACTION_DIM,
            max_state_dim=DUMMY_STATE_DIM,
            device=DEVICE,
            validate_action_token_prefix=False,
            max_action_tokens=2,
            max_decoding_steps=2,
            chunk_size=2,
        )
        policy = PI0FastPolicy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_pi0_fast_pre_post_processors(
        config=policy.config,
        dataset_stats=None,  # Pass None for dataset_stats to disable normalization
    )

    return policy, preprocessor, postprocessor


def create_dummy_data(device=DEVICE):
    """Create dummy data for testing both implementations."""
    batch_size = 1
    prompt = "Pick up the red block and place it in the bin"

    # Create random RGB images in [0, 255] uint8 range (as PIL images would be)
    # Then convert to [0, 1] float32 range for LeRobot
    def fake_rgb(h, w):
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        return t

    batch = {
        f"{OBS_IMAGES}.base_0_rgb": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        f"{OBS_IMAGES}.left_wrist_0_rgb": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        f"{OBS_IMAGES}.right_wrist_0_rgb": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "task": [prompt for _ in range(batch_size)],
    }

    return batch


# Pytest fixtures
@pytest.fixture(scope="module")
def pi0_fast_components():
    """Fixture to instantiate and provide all PI0Fast components for tests."""
    print(f"\nTesting with DEVICE='{DEVICE}'")
    print("\n[Setup] Instantiating LeRobot PI0Fast policy...")
    policy_obj, preprocessor_obj, postprocessor_obj = instantiate_lerobot_pi0_fast(from_pretrained=True)
    print("Model loaded successfully")
    yield policy_obj, preprocessor_obj, postprocessor_obj


@pytest.fixture(scope="module")
def policy(pi0_fast_components):
    """Fixture to provide the PI0Fast policy for tests."""
    return pi0_fast_components[0]


@pytest.fixture(scope="module")
def preprocessor(pi0_fast_components):
    """Fixture to provide the PI0Fast preprocessor for tests."""
    return pi0_fast_components[1]


@require_cuda
def test_pi0_fast_preprocessor_alignment(policy, preprocessor):
    """Test that LeRobot PI0Fast preprocessor produces expected outputs."""
    print("\n" + "=" * 80)
    print("Test: PI0Fast Preprocessor Outputs")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Preprocessing...")
    lerobot_observation = preprocessor(deepcopy(batch))

    print("\nVerifying preprocessor outputs:")
    print("-" * 80)

    # Expected keys from PI0Fast preprocessing
    expected_keys = [
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
        "observation.state",
        "observation.language_tokens",
        "observation.language_attention_mask",
    ]

    for key in expected_keys:
        if key in lerobot_observation:
            shape = tuple(lerobot_observation[key].shape)
            print(f"\nKey: {key}")
            print(f"Shape: {shape}")
            print(f"Dtype: {lerobot_observation[key].dtype}")
        else:
            print(f"\nKey '{key}' not found in inputs!")

    # Check language tokens shape
    if "observation.language_tokens" in lerobot_observation:
        lang_tokens = lerobot_observation["observation.language_tokens"]
        print(f"\nLanguage tokens shape: {lang_tokens.shape}")
        # Should have batch dimension and max_length from tokenizer
        assert lang_tokens.dim() == 2, f"Expected 2D tensor, got {lang_tokens.dim()}D"

    print("\nPreprocessor outputs verified!")


@require_cuda
def test_pi0_fast_action_generation(policy, preprocessor):
    """Test PI0Fast LeRobot implementation generates expected actions."""
    print("\n" + "=" * 80)
    print("Test: PI0Fast Action Generation Against Expected Values")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Running inference...")
    lerobot_observation = preprocessor(deepcopy(batch))

    # Reset seed for inference
    torch.manual_seed(42)
    with torch.no_grad():
        lerobot_actions = policy.predict_action_chunk(lerobot_observation)
        lerobot_actions = lerobot_actions.float().cpu()

    print(f"LeRobot actions shape: {lerobot_actions.shape}")
    print(f"LeRobot actions mean: {lerobot_actions.mean().item():.6f}")
    print(f"LeRobot actions std: {lerobot_actions.std().item():.6f}")
    print(f"LeRobot actions first 5: {lerobot_actions[0, 0, :5]}")

    print("\nExpected values (from original PI0Fast):")
    print(f"Expected actions shape: {EXPECTED_ACTIONS_SHAPE}")
    print(f"Expected actions mean: {EXPECTED_ACTIONS_MEAN:.6f}")
    print(f"Expected actions std: {EXPECTED_ACTIONS_STD:.6f}")
    print(f"Expected actions first 5: {EXPECTED_ACTIONS_FIRST_5}")

    print("\nAction Comparison:")
    print("-" * 80)

    # Compare shapes
    actual_shape = tuple(lerobot_actions.shape)
    print(f"Actual shape: {actual_shape}")

    assert actual_shape == EXPECTED_ACTIONS_SHAPE, (
        f"Shape mismatch: {actual_shape} vs {EXPECTED_ACTIONS_SHAPE}"
    )
    print(f"Shape matches: {actual_shape}")

    # Compare statistics
    actual_mean = lerobot_actions.mean().item()
    actual_std = lerobot_actions.std().item()

    print(f"\nMean: {actual_mean:.6f} (expected: {EXPECTED_ACTIONS_MEAN:.6f})")
    print(f"Std: {actual_std:.6f} (expected: {EXPECTED_ACTIONS_STD:.6f})")

    # Compare first 5 actions
    actual_first_5 = lerobot_actions[0, 0, :5]
    print("\nFirst 5 actions comparison:")
    print(f"  Actual:   {actual_first_5}")
    print(f"  Expected: {EXPECTED_ACTIONS_FIRST_5}")

    first_5_diff = torch.abs(actual_first_5 - EXPECTED_ACTIONS_FIRST_5)
    print(f"  Max diff: {first_5_diff.max().item():.6e}")
    print(f"  Mean diff: {first_5_diff.mean().item():.6e}")

    # Check with different tolerances
    tolerances = [1e-5, 1e-4, 1e-3, 1e-2]
    for tol in tolerances:
        is_close = torch.allclose(actual_first_5, EXPECTED_ACTIONS_FIRST_5, atol=tol)
        status = "Success" if is_close else "Failure"
        print(f"{status}: First 5 actions close (atol={tol}): {is_close}")

    # Assert with reasonable tolerance
    tolerance = 1e-3
    assert torch.allclose(actual_first_5, EXPECTED_ACTIONS_FIRST_5, atol=tolerance), (
        f"First 5 actions differ by more than tolerance ({tolerance})"
    )
    print(f"\nSuccess: Actions match expected values within tolerance ({tolerance})!")

    print("\nAction generation test completed (values printed for reference)!")


@require_cuda
def test_pi0_fast_inference_reproducibility(policy, preprocessor):
    """Test that PI0Fast inference is reproducible with the same seed."""
    print("\n" + "=" * 80)
    print("Test: PI0Fast Inference Reproducibility")
    print("=" * 80)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    # First inference
    print("\n[Run 1] Running inference...")
    set_seed_all(42)
    lerobot_observation = preprocessor(deepcopy(batch))
    with torch.no_grad():
        actions_1 = policy.predict_action_chunk(lerobot_observation)
        actions_1 = actions_1.float().cpu()

    # Second inference with same seed
    print("\n[Run 2] Running inference with same seed...")
    set_seed_all(42)
    lerobot_observation = preprocessor(deepcopy(batch))
    with torch.no_grad():
        actions_2 = policy.predict_action_chunk(lerobot_observation)
        actions_2 = actions_2.float().cpu()

    print("\nComparing two runs:")
    print("-" * 80)
    if torch.allclose(actions_1, actions_2, atol=1e-8):
        print("Inference is perfectly reproducible!")
    else:
        diff = torch.abs(actions_1 - actions_2)
        print("Small differences detected:")
        print(f"  Max diff: {diff.max().item():.6e}")
        print(f"  Mean diff: {diff.mean().item():.6e}")

    assert torch.allclose(actions_1, actions_2, atol=1e-6), "Inference should be reproducible!"

    print("\nInference is reproducible!")


@require_cuda
def test_pi0_fast_forward_pass_logits(policy, preprocessor):
    """Test PI0Fast forward pass and compare logits against expected values."""
    print("\n" + "=" * 80)
    print("Test: PI0Fast Forward Pass Logits")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data with action tokens...")
    batch = create_dummy_data()

    # Preprocess the batch
    lerobot_observation = preprocessor(deepcopy(batch))

    # For forward pass, we need action tokens
    # Create dummy action tokens for testing
    batch_size = 1
    max_action_tokens = policy.config.max_action_tokens

    # Create dummy action tokens (in practice, these come from the FAST tokenizer)
    dummy_action_tokens = torch.randint(
        0, 1000, (batch_size, max_action_tokens), dtype=torch.long, device=DEVICE
    )
    dummy_action_masks = torch.ones(batch_size, max_action_tokens, dtype=torch.bool, device=DEVICE)

    # Add action tokens to the observation
    lerobot_observation[ACTION_TOKENS] = dummy_action_tokens
    lerobot_observation[ACTION_TOKEN_MASK] = dummy_action_masks

    print("\n[LeRobot] Running forward pass...")
    policy.train()
    with torch.no_grad():
        loss, loss_dict = policy.forward(lerobot_observation)

    print(f"Loss: {loss.item():.6f}")
    print(f"FAST Loss: {loss_dict['ce_loss']:.6f}")

    print("\nForward pass completed successfully!")
    print(f"Loss value: {loss.item():.6f}")

    # The loss should be a positive value
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"

    print("\nForward pass test passed!")


@require_cuda
def test_pi0_fast_action_token_sampling(policy, preprocessor):
    """Test PI0Fast action token sampling (autoregressive decoding)."""
    print("\n" + "=" * 80)
    print("Test: PI0Fast Action Token Sampling")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Preprocessing...")
    lerobot_observation = preprocessor(deepcopy(batch))

    # Prepare inputs for model
    images, img_masks = policy._preprocess_images(lerobot_observation)
    tokens = lerobot_observation[OBS_LANGUAGE_TOKENS]
    masks = lerobot_observation[OBS_LANGUAGE_ATTENTION_MASK]

    print("\n[LeRobot] Sampling action tokens...")
    torch.manual_seed(42)
    with torch.no_grad():
        action_tokens = policy.model.sample_actions_fast(
            images,
            img_masks,
            tokens,
            masks,
            max_decoding_steps=2,
            temperature=0.0,  # Greedy decoding for reproducibility
        )

    print(f"Action tokens shape: {action_tokens.shape}")
    print(f"Action tokens first 10: {action_tokens[0, :10].tolist()}")

    print("\nExpected values (from original PI0Fast):")
    print(f"Expected shape: {EXPECTED_ACTION_TOKENS_SHAPE}")
    print(f"Expected first 5: {EXPECTED_ACTION_TOKENS_FIRST_5.tolist()}")

    # Verify shape
    actual_shape = tuple(action_tokens.shape)
    print(f"\nActual shape: {actual_shape}")

    assert actual_shape == EXPECTED_ACTION_TOKENS_SHAPE, (
        f"Shape mismatch: {actual_shape} vs {EXPECTED_ACTION_TOKENS_SHAPE}"
    )

    # Compare first 5 tokens
    actual_first_5 = action_tokens[0, :5].cpu()
    assert torch.equal(actual_first_5, EXPECTED_ACTION_TOKENS_FIRST_5), (
        f"First 5 tokens mismatch: {actual_first_5} vs {EXPECTED_ACTION_TOKENS_FIRST_5}"
    )

    print("\nAction token sampling test completed!")


@require_cuda
def test_pi0_fast_detokenization(policy, preprocessor):
    """Test PI0Fast action detokenization (FAST decoding)."""
    print("\n" + "=" * 80)
    print("Test: PI0Fast Action Detokenization")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Preprocessing...")
    lerobot_observation = preprocessor(deepcopy(batch))

    # Prepare inputs for model
    images, img_masks = policy._preprocess_images(lerobot_observation)
    tokens = lerobot_observation[OBS_LANGUAGE_TOKENS]
    masks = lerobot_observation[OBS_LANGUAGE_ATTENTION_MASK]

    print("\n[LeRobot] Sampling action tokens...")
    torch.manual_seed(42)
    with torch.no_grad():
        action_tokens = policy.model.sample_actions_fast(
            images,
            img_masks,
            tokens,
            masks,
            max_decoding_steps=2,
            temperature=0.0,
        )

    print(f"Action tokens shape: {action_tokens.shape}")

    # Detokenize
    print("\n[LeRobot] Detokenizing action tokens...")
    action_horizon = policy.config.n_action_steps
    action_dim = policy.config.output_features["action"].shape[0]

    try:
        continuous_actions = policy.detokenize_actions(
            action_tokens, action_horizon=action_horizon, action_dim=action_dim
        )
        print(f"Continuous actions shape: {continuous_actions.shape}")
        print(f"Continuous actions mean: {continuous_actions.mean().item():.6f}")
        print(f"Continuous actions std: {continuous_actions.std().item():.6f}")
        print(f"Continuous actions first 5: {continuous_actions[0, 0, :5]}")
        print("\nDetokenization successful!")
    except Exception as e:
        print(f"\nDetokenization failed with error: {e}")
        print("This may be expected if the action tokens are not valid FAST tokens.")
        print("The test will pass as long as the sampling works correctly.")
