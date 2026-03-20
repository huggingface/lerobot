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

"""Tests for the reward model base classes and registry."""

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel


def test_reward_model_config_registry():
    """Verify that classifier and sarm are registered."""
    known = RewardModelConfig.get_known_choices()
    assert "reward_classifier" in known
    assert "sarm" in known


def test_reward_model_config_lookup():
    """Verify that we can look up configs by name."""
    cls = RewardModelConfig.get_choice_class("reward_classifier")
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig

    assert cls is RewardClassifierConfig


def test_factory_get_reward_model_class():
    """Test the get_reward_model_class factory."""
    from lerobot.rewards.factory import get_reward_model_class

    cls = get_reward_model_class("sarm")
    from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel

    assert cls is SARMRewardModel


def test_factory_unknown_raises():
    """Unknown name should raise ValueError."""
    from lerobot.rewards.factory import get_reward_model_class

    unknown_type = "nonexistent_reward_model"
    with pytest.raises(ValueError, match=f"Unknown reward model type '{unknown_type}'") as exc_info:
        get_reward_model_class(unknown_type)

    message = str(exc_info.value)
    assert "Available reward models:" in message
    assert "reward_classifier" in message
    assert "sarm" in message


def test_pretrained_reward_model_requires_config_class():
    """Subclass without config_class should fail."""
    with pytest.raises(TypeError, match="must define 'config_class'"):

        class BadModel(PreTrainedRewardModel):
            name = "bad"

            def compute_reward(self, batch):
                pass


def test_pretrained_reward_model_requires_name():
    """Subclass without name should fail."""
    with pytest.raises(TypeError, match="must define 'name'"):

        class BadModel(PreTrainedRewardModel):
            config_class = RewardModelConfig

            def compute_reward(self, batch):
                pass


def test_non_trainable_forward_raises():
    """Non-trainable model should raise on forward()."""
    from dataclasses import dataclass

    from lerobot.optim.optimizers import AdamWConfig

    @dataclass
    class DummyConfig(RewardModelConfig):
        def get_optimizer_preset(self):
            return AdamWConfig(lr=1e-4)

    class DummyReward(PreTrainedRewardModel):
        config_class = DummyConfig
        name = "dummy_test"

        def compute_reward(self, batch):
            return torch.zeros(1)

    config = DummyConfig()
    model = DummyReward(config)

    with pytest.raises(NotImplementedError, match="not trainable"):
        model.forward({"x": torch.zeros(1)})
