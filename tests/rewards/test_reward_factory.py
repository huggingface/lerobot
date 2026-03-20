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

"""Tests for reward factory error messages and unknown types."""

import pytest

from lerobot.rewards.factory import (
    get_reward_model_class,
    make_reward_model_config,
    make_reward_pre_post_processors,
)


def test_get_reward_model_class_unknown_lists_available_choices():
    unknown_type = "nonexistent_reward_model"

    with pytest.raises(ValueError, match=f"Unknown reward model type '{unknown_type}'") as exc_info:
        get_reward_model_class(unknown_type)

    message = str(exc_info.value)
    assert "Available reward models:" in message
    assert "reward_classifier" in message
    assert "sarm" in message


def test_make_reward_model_config_unknown_lists_available_choices():
    unknown_type = "nonexistent_reward_model"

    with pytest.raises(ValueError, match=f"Unknown reward model type '{unknown_type}'") as exc_info:
        make_reward_model_config(unknown_type)

    message = str(exc_info.value)
    assert "Available reward models:" in message
    assert "reward_classifier" in message
    assert "sarm" in message


def test_make_reward_pre_post_processors_unknown_lists_available_choices():
    class DummyUnsupportedRewardConfig:
        type = "unknown_reward_type_for_processor"

    with pytest.raises(
        ValueError, match="Processor for reward model type 'unknown_reward_type_for_processor' is not implemented."
    ) as exc_info:
        make_reward_pre_post_processors(DummyUnsupportedRewardConfig())

    message = str(exc_info.value)
    assert "Available reward models:" in message
    assert "reward_classifier" in message
    assert "sarm" in message
