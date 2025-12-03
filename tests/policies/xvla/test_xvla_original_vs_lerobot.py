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

"""Test script to verify XVLA policy integration with LeRobot vs the original implementation, only meant to be run locally!"""
# ruff: noqa: E402

import random
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402
from tests.utils import require_cuda  # noqa: E402

# Constants
DUMMY_ACTION_DIM = 7  # Standard robot arm action dimension
DUMMY_STATE_DIM = 20  # Proprioceptive state dimension
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_VIEWS = 2  # Number of camera views
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH_LEROBOT = "lerobot/xvla-widowx"
LIBERO_DOMAIN_ID = 0  # Domain ID for examples purposes

# Expected values from original XVLA implementation (reference values)
EXPECTED_ACTIONS_SHAPE = (30, 20)
EXPECTED_ACTIONS_MEAN = 0.117606
EXPECTED_ACTIONS_STD = 0.245411
EXPECTED_ACTIONS_FIRST_5 = torch.tensor([0.2742, 0.4977, 0.0500, 0.7040, -0.2653])


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


def instantiate_lerobot_xvla(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH_LEROBOT,
) -> tuple[
    Any,  # Policy
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Instantiate LeRobot XVLA policy with preprocessor and postprocessor."""
    if from_pretrained:
        policy = XVLAPolicy.from_pretrained(
            pretrained_name_or_path=model_path,
            strict=False,
        )
    else:
        config = XVLAConfig(
            base_model_path=model_path,
            n_action_steps=DUMMY_ACTION_DIM,
            chunk_size=DUMMY_ACTION_DIM,
            device=DEVICE,
            num_image_views=NUM_VIEWS,
        )  # add resize_imgs_with_padding=IMAGE_SIZE, IMAGE_SIZE?
        policy = XVLAPolicy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_xvla_pre_post_processors(
        config=policy.config,
        dataset_stats=None,  # Pass None for dataset_stats to disable normalization (original XVLA doesn't normalize)
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
        f"{OBS_IMAGES}.image": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        f"{OBS_IMAGES}.image2": torch.stack(
            [fake_rgb(IMAGE_HEIGHT, IMAGE_WIDTH) for _ in range(batch_size)]
        ).to(device),
        OBS_STATE: torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "task": [prompt for _ in range(batch_size)],
    }

    return batch


# Pytest fixtures
@pytest.fixture(scope="module")
def xvla_components():
    """Fixture to instantiate and provide all XVLA components for tests."""
    print(f"\nTesting with DEVICE='{DEVICE}'")
    print("\n[Setup] Instantiating LeRobot XVLA policy...")
    policy_obj, preprocessor_obj, postprocessor_obj = instantiate_lerobot_xvla(from_pretrained=True)
    print("✔️ Model loaded successfully")
    yield policy_obj, preprocessor_obj, postprocessor_obj


@pytest.fixture(scope="module")
def policy(xvla_components):
    """Fixture to provide the XVLA policy for tests."""
    return xvla_components[0]


@pytest.fixture(scope="module")
def preprocessor(xvla_components):
    """Fixture to provide the XVLA preprocessor for tests."""
    return xvla_components[1]


@require_cuda
def test_xvla_preprocessor_alignment(policy, preprocessor):
    """Test that LeRobot XVLA preprocessor produces expected outputs."""
    print("\n" + "=" * 80)
    print("Test: XVLA Preprocessor Outputs")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Preprocessing...")
    lerobot_observation = preprocessor(deepcopy(batch))
    lerobot_inputs = policy._build_model_inputs(lerobot_observation)

    print("\nVerifying preprocessor outputs:")
    print("-" * 80)

    # Expected shapes from tester.txt
    expected_shapes = {
        "domain_id": (1,),
        "input_ids": (1, 50),
        "proprio": (1, 20),
        "image_mask": (1, 2),
        "image_input": (1, 2, 3, 224, 224),
    }

    for key, expected_shape in expected_shapes.items():
        if key in lerobot_inputs:
            actual_shape = tuple(lerobot_inputs[key].shape)
            print(f"\nKey: {key}")
            print(f"Expected shape: {expected_shape}")
            print(f"Actual shape: {actual_shape}")

            if actual_shape == expected_shape:
                print("Shape matches!")
            else:
                print("Shape mismatch!")

            assert actual_shape == expected_shape, f"Shape mismatch for {key}"
        else:
            print(f"\nKey '{key}' not found in inputs!")

    print("\nAll preprocessor outputs have correct shapes!")


@require_cuda
def test_xvla_action_generation(policy, preprocessor):
    """Test XVLA LeRobot implementation generates expected actions."""
    print("\n" + "=" * 80)
    print("Test: XVLA Action Generation Against Expected Values")
    print("=" * 80)

    set_seed_all(42)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    print("\n[LeRobot] Running inference...")
    lerobot_observation = preprocessor(deepcopy(batch))
    lerobot_inputs = policy._build_model_inputs(lerobot_observation)

    # Reset seed for inference
    torch.manual_seed(42)
    with torch.no_grad():
        lerobot_actions = policy.model.generate_actions(**lerobot_inputs, steps=10)
        lerobot_actions = lerobot_actions.squeeze(0).float().cpu()

    print(f"LeRobot actions shape: {lerobot_actions.shape}")
    print(f"LeRobot actions mean: {lerobot_actions.mean().item():.6f}")
    print(f"LeRobot actions std: {lerobot_actions.std().item():.6f}")
    print(f"LeRobot actions first 5: {lerobot_actions[0, :5]}")

    print("\nExpected values (from original XVLA):")
    print(f"Expected actions shape: {EXPECTED_ACTIONS_SHAPE}")
    print(f"Expected actions mean: {EXPECTED_ACTIONS_MEAN:.6f}")
    print(f"Expected actions std: {EXPECTED_ACTIONS_STD:.6f}")
    print(f"Expected actions first 5: {EXPECTED_ACTIONS_FIRST_5}")

    print("\nAction Comparison:")
    print("-" * 80)

    # Compare shapes
    actual_shape = tuple(lerobot_actions.shape)
    assert actual_shape == EXPECTED_ACTIONS_SHAPE, (
        f"Shape mismatch: {actual_shape} vs {EXPECTED_ACTIONS_SHAPE}"
    )
    print(f"✔️ Shape matches: {actual_shape}")

    # Compare statistics
    actual_mean = lerobot_actions.mean().item()
    actual_std = lerobot_actions.std().item()

    mean_diff = abs(actual_mean - EXPECTED_ACTIONS_MEAN)
    std_diff = abs(actual_std - EXPECTED_ACTIONS_STD)

    print(f"\nMean: {actual_mean:.6f} (expected: {EXPECTED_ACTIONS_MEAN:.6f}, diff: {mean_diff:.6e})")
    print(f"Std: {actual_std:.6f} (expected: {EXPECTED_ACTIONS_STD:.6f}, diff: {std_diff:.6e})")

    # Compare first 5 actions
    actual_first_5 = lerobot_actions[0, :5]
    first_5_diff = torch.abs(actual_first_5 - EXPECTED_ACTIONS_FIRST_5)

    print("\nFirst 5 actions comparison:")
    print(f"  Actual:   {actual_first_5}")
    print(f"  Expected: {EXPECTED_ACTIONS_FIRST_5}")
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


@require_cuda
def test_xvla_inference_reproducibility(policy, preprocessor):
    """Test that XVLA inference is reproducible with the same seed."""
    print("\n" + "=" * 80)
    print("Test: XVLA Inference Reproducibility")
    print("=" * 80)

    print("\nCreating dummy data...")
    batch = create_dummy_data()

    # First inference
    print("\n[Run 1] Running inference...")
    set_seed_all(42)
    lerobot_observation = preprocessor(deepcopy(batch))
    lerobot_inputs = policy._build_model_inputs(lerobot_observation)
    with torch.no_grad():
        actions_1 = policy.model.generate_actions(**lerobot_inputs, steps=10)
        actions_1 = actions_1.squeeze(0).float().cpu()

    # Second inference with same seed
    print("\n[Run 2] Running inference with same seed...")
    set_seed_all(42)
    lerobot_observation = preprocessor(deepcopy(batch))
    lerobot_inputs = policy._build_model_inputs(lerobot_observation)
    with torch.no_grad():
        actions_2 = policy.model.generate_actions(**lerobot_inputs, steps=10)
        actions_2 = actions_2.squeeze(0).float().cpu()

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
