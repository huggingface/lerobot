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

"""Test script for LeRobot's Groot policy forward and inference passes."""

import gc
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.utils import auto_select_torch_device
from tests.utils import require_cuda  # noqa: E402

pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local Groot installation and is not meant for CI",
)


# Define constants for dummy data
DUMMY_STATE_DIM = 44
DUMMY_ACTION_DIM = 44
DUMMY_ACTION_HORIZON = 16
IMAGE_SIZE = 256
DEVICE = auto_select_torch_device()
MODEL_PATH = "aractingi/bimanual-handover-groot-10k"


def cleanup_memory():
    """Clean up GPU/MPS memory to prevent OOM errors between tests."""
    print("\nCleaning up memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Memory cleanup complete.")


def set_seed_all(seed: int):
    """Set random seed for all RNG sources to ensure reproducibility."""
    import random

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


def instantiate_lerobot_groot(
    from_pretrained: bool = False,
    model_path: str = MODEL_PATH,
) -> tuple[
    GrootPolicy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Instantiate LeRobot Groot policy with preprocessor and postprocessor."""
    if from_pretrained:
        policy = GrootPolicy.from_pretrained(
            pretrained_name_or_path=model_path,
            strict=False,
        )
        policy.config.embodiment_tag = "gr1"
    else:
        config = GrootConfig(
            base_model_path=model_path,
            n_action_steps=DUMMY_ACTION_HORIZON,
            chunk_size=DUMMY_ACTION_HORIZON,
            image_size=[IMAGE_SIZE, IMAGE_SIZE],
            device=DEVICE,
            embodiment_tag="gr1",
        )
        policy = GrootPolicy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE

    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=policy.config,
        dataset_stats=None,  # Pass None for dataset_stats to disable normalization (original GR00T doesn't normalize)
    )

    return (policy, preprocessor, postprocessor)


def create_dummy_data(device=DEVICE):
    """Create a dummy data batch for testing."""
    batch_size = 2
    prompt = "Pick up the red cube and place it in the bin"
    state = torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device)

    batch = {
        "observation.state": state,
        "action": torch.randn(
            batch_size,
            DUMMY_ACTION_HORIZON,
            DUMMY_ACTION_DIM,
            dtype=torch.float32,
            device=device,  # Action ground truth (for training)
        ),
        "observation.images.ego_view": torch.rand(
            batch_size,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            dtype=torch.float32,
            device=device,  # Images in [0, 1] range as expected by LeRobot
        ),
        "task": [prompt for _ in range(batch_size)],
    }

    return batch


@require_cuda
def test_lerobot_groot_inference():
    """Test the inference pass (select_action) of LeRobot's Groot policy."""
    print("Test: LeRobot Groot Inference Pass")

    set_seed_all(42)

    # Instantiate policy and processors
    lerobot_policy, lerobot_preprocessor, lerobot_postprocessor = instantiate_lerobot_groot(
        from_pretrained=True
    )
    batch = create_dummy_data()

    print("\n[LeRobot] Running inference...")
    lerobot_policy.eval()
    batch_lerobot_processed = lerobot_preprocessor(deepcopy(batch))

    # Ensure identical RNG state before inference
    torch.manual_seed(42)

    with torch.no_grad():
        lerobot_action = lerobot_policy.select_action(batch_lerobot_processed)

    print(f"\nInference successful. Output action shape: {lerobot_action.shape}")
    print("Output actions (first 5 dims):")
    print(lerobot_action[:, :5])

    lerobot_action = lerobot_postprocessor(lerobot_action)

    del lerobot_policy, lerobot_preprocessor, lerobot_postprocessor, batch
    cleanup_memory()


@require_cuda
def test_lerobot_groot_forward_pass():
    """Test the forward pass of LeRobot's Groot policy."""
    print("\n" + "=" * 50)
    print("Test: LeRobot Groot Forward Pass (Training Mode)")

    set_seed_all(42)

    # Instantiate policy and processors
    lerobot_policy, lerobot_preprocessor, _ = instantiate_lerobot_groot(from_pretrained=True)
    batch = create_dummy_data()

    lerobot_policy.eval()

    print("\n[LeRobot] Running forward pass...")
    batch_lerobot_processed = lerobot_preprocessor(deepcopy(batch))

    set_seed_all(42)
    with torch.no_grad():
        lerobot_loss, lerobot_metrics = lerobot_policy.forward(batch_lerobot_processed)

    print("\nForward pass successful.")
    print(f"  - Loss: {lerobot_loss.item():.6f}")
    print(f"  - Metrics: {lerobot_metrics}")

    del lerobot_policy, lerobot_preprocessor, batch
    cleanup_memory()
