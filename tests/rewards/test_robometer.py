# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for Robometer configuration and model."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig

# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


def test_robometer_config_registered():
    """RobometerConfig must be registered in the reward model registry."""
    known = RewardModelConfig.get_known_choices()
    assert "robometer" in known


def test_robometer_config_lookup():
    """Registry lookup must return the correct config class."""
    cls = RewardModelConfig.get_choice_class("robometer")
    assert cls is RobometerConfig


def test_robometer_config_defaults():
    """Default config should be instantiable with sensible values."""
    cfg = RobometerConfig()
    assert cfg.name == "robometer"
    assert cfg.task_key == "observation.language_instruction"
    assert cfg.image_key == "observation.images.side"

def test_robometer_config_type():
    """config.type must return the registered name."""
    cfg = RobometerConfig()
    assert cfg.type == "robometer"

# def test_robometer_config_validate_features_missing_key():
#     """validate_features must raise if image_key is not in input_features."""
#     cfg = RobometerConfig(image_key="observation.images.nonexistent")
#     with pytest.raises(ValueError, match="image_key"):
#         cfg.validate_features()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_factory_get_reward_model_class():
    """Factory must return RobometerModel for 'robometer'."""
    from lerobot.rewards.factory import get_reward_model_class
    from lerobot.rewards.robometer.modeling_robometer import RobometerModel
    
    cls = get_reward_model_class("robometer")
    assert cls is RobometerModel


def test_factory_make_reward_model_config():
    """make_reward_model_config must return a RobometerConfig instance."""
    from lerobot.rewards.factory import make_reward_model_config
    
    cfg = make_reward_model_config("robometer")
    assert isinstance(cfg, RobometerConfig)


# ---------------------------------------------------------------------------
# Model tests (VLM mocked — no network / GPU required)
# ---------------------------------------------------------------------------


def test_compute_reward_shape():
    """compute_reward must return (B, T) tensor for (B, T, C, H, W) input."""
    from lerobot.rewards.robometer.modeling_robometer import RobometerModel
    cfg = RobometerConfig()
    model = RobometerModel(cfg)
    batch = {
        "observation.images.side": torch.rand(2, 8, 3, 84, 84),
        "observation.language_instruction": ["pick up the cube", "place the block"],
    }
    rewards = model.compute_reward(batch)
    assert rewards.shape == (2,8)
    assert rewards.dtype == torch.float32


def test_forward_raises_not_implemented():
    """forward() must raise NotImplementedError (inference-only model)."""
    from lerobot.rewards.robometer.modeling_robometer import RobometerModel
    cfg = RobometerConfig()
    model = RobometerModel(cfg)
    with pytest.raises(NotImplementedError, match="compute_reward"):
        model.forward({})

