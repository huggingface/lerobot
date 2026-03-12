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

"""Test script to verify Wall-X policy integration with LeRobot, only meant to be run locally!"""

import os

import pytest
import torch

# Skip if required dependencies are not available
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("torchdiffeq")

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local Wall-X installation and is not meant for CI",
)

from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.wall_x import WallXConfig  # noqa: E402
from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy  # noqa: E402
from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402


def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = WallXConfig(device="cuda")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(7,),
        ),
        "observation.images.face_view": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "observation.images.face_view": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = WallXPolicy(config)
    preprocessor, postprocessor = make_wall_x_pre_post_processors(config=config, dataset_stats=dataset_stats)
    # Test forward pass with dummy data
    batch_size = 1
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 7, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.face_view": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    # Test inference
    batch = {
        "observation.state": torch.randn(batch_size, 7, dtype=torch.float32, device=device),
        "observation.images.face_view": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise


def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="wall_x",
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise


if __name__ == "__main__":
    test_policy_instantiation()
    test_config_creation()
