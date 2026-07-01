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

"""Test script to verify LingBot-VLA policy integration with LeRobot."""

import pytest
import torch

# Skip if required dependencies are not available
pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.lingbot_vla import LingbotVLAConfig  # noqa: E402
from lerobot.policies.lingbot_vla.modeling_lingbot_vla import LingbotVLAPolicy  # noqa: E402
from lerobot.policies.lingbot_vla.processor_lingbot_vla import (  # noqa: E402
    make_lingbot_vla_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda  # noqa: E402

STATE_DIM = 6
ACTION_DIM = 6
CAM = "observation.images.cam"


def _make_config() -> LingbotVLAConfig:
    config = LingbotVLAConfig(device="cuda", attention_implementation="eager")
    config.input_features = {
        CAM: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return config


def _make_dataset_stats() -> dict:
    return {
        OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        ACTION: {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
    }


@require_cuda
def test_policy_instantiation():
    """Raw observation -> processor pipeline -> forward + select_action round-trip."""
    set_seed(42)
    config = _make_config()
    dataset_stats = _make_dataset_stats()

    policy = LingbotVLAPolicy(config).to(device="cuda", dtype=torch.bfloat16)
    policy.eval()
    preprocessor, postprocessor = make_lingbot_vla_pre_post_processors(
        config=config, dataset_stats=dataset_stats
    )

    # Forward (training) pass: a pre-batched sample as it would arrive from the
    # dataloader (batch dim already present on image/state/action).
    train_obs = {
        CAM: torch.rand(1, 3, 480, 640),
        OBS_STATE: torch.randn(1, STATE_DIM),
        ACTION: torch.randn(1, config.chunk_size, ACTION_DIM),
        "task": "pick up the red cube",
    }
    batch = preprocessor(train_obs)
    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss), loss
    assert "loss" in loss_dict

    # Inference pass: select a single action and unnormalize it.
    infer_obs = {
        CAM: torch.rand(3, 480, 640),
        OBS_STATE: torch.randn(STATE_DIM),
        "task": "pick up the red cube",
    }
    batch = preprocessor(infer_obs)
    with torch.no_grad():
        action = policy.select_action(batch)
        action = postprocessor(action)
    assert action.shape[-1] == ACTION_DIM, action.shape


@require_cuda
def test_processor_produces_model_ready_keys():
    """The preprocessor must emit the patchified images/lang tensors the model consumes."""
    config = _make_config()
    preprocessor, _ = make_lingbot_vla_pre_post_processors(config=config, dataset_stats=_make_dataset_stats())
    obs = {
        CAM: torch.rand(3, 480, 640),
        OBS_STATE: torch.randn(STATE_DIM),
        "task": "pick up the red cube",
    }
    batch = preprocessor(obs)
    for key in ("images", "img_masks", "lang_tokens", "lang_masks", OBS_STATE):
        assert key in batch, key
    assert batch["images"].ndim == 4  # (B, n_views, num_patches, patch_dim)
    assert batch["lang_tokens"].shape[-1] == config.tokenizer_max_length


def test_config_creation():
    """Config can be created through the policy factory by type name."""
    config = make_policy_config(policy_type="lingbot_vla")
    assert type(config).__name__ == "LingbotVLAConfig"
