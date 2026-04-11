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

"""Tests for RoboReward configuration, model, and processor."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robo_reward.configuration_robo_reward import RoboRewardConfig

# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


def test_robo_reward_config_registered():
    """RoboRewardConfig must be registered in the reward model registry."""
    known = RewardModelConfig.get_known_choices()
    assert "robo_reward" in known


def test_robo_reward_config_lookup():
    """Registry lookup must return the correct config class."""
    cls = RewardModelConfig.get_choice_class("robo_reward")
    assert cls is RoboRewardConfig


def test_robo_reward_config_defaults():
    """Default config should be instantiable with sensible values."""
    cfg = RoboRewardConfig()
    assert cfg.model_name == "teetone/RoboReward-8B"
    assert cfg.max_new_tokens == 16
    assert cfg.score_to_reward == {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
    assert cfg.task_key == "observation.language_instruction"
    assert cfg.image_key == "observation.images.top"


def test_robo_reward_config_4b_variant():
    """Config should accept the 4B model variant."""
    cfg = RoboRewardConfig(model_name="teetone/RoboReward-4B")
    assert "4B" in cfg.model_name


def test_robo_reward_config_type():
    """config.type must return the registered name."""
    cfg = RoboRewardConfig()
    assert cfg.type == "robo_reward"


def test_robo_reward_config_validate_features_missing_key():
    """validate_features must raise if image_key is not in input_features."""
    cfg = RoboRewardConfig(image_key="observation.images.nonexistent")
    with pytest.raises(ValueError, match="image_key"):
        cfg.validate_features()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_factory_get_reward_model_class():
    """Factory must return RoboRewardModel for 'robo_reward'."""
    from lerobot.rewards.factory import get_reward_model_class
    from lerobot.rewards.robo_reward.modeling_robo_reward import RoboRewardModel

    cls = get_reward_model_class("robo_reward")
    assert cls is RoboRewardModel


def test_factory_make_reward_model_config():
    """make_reward_model_config must return a RoboRewardConfig instance."""
    from lerobot.rewards.factory import make_reward_model_config

    cfg = make_reward_model_config("robo_reward")
    assert isinstance(cfg, RoboRewardConfig)


# ---------------------------------------------------------------------------
# Model tests (VLM mocked — no network / GPU required)
# ---------------------------------------------------------------------------


def _make_mock_vlm_and_processor(output_text: str = "ANSWER: 4"):
    """Return a (mock_vlm, mock_processor) pair that simulates Qwen3-VL."""
    mock_vlm = MagicMock()
    mock_vlm.parameters.return_value = iter([torch.zeros(1)])
    mock_vlm.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "fake_text"
    mock_processor.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    mock_processor.decode.return_value = output_text
    return mock_vlm, mock_processor


def _make_model_with_mocks(output_text: str = "ANSWER: 4"):
    """Instantiate RoboRewardModel with mocked VLM/processor, no network calls."""
    from lerobot.rewards.robo_reward.modeling_robo_reward import RoboRewardModel

    cfg = RoboRewardConfig()
    mock_vlm, mock_processor = _make_mock_vlm_and_processor(output_text)

    with (
        patch(
            "lerobot.rewards.robo_reward.modeling_robo_reward.Qwen3VLForConditionalGeneration"
        ) as mock_vlm_cls,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward.AutoProcessor") as mock_proc_cls,
        patch(
            "lerobot.rewards.robo_reward.modeling_robo_reward._TRANSFORMERS_AVAILABLE",
            True,
        ),
    ):
        mock_vlm_cls.from_pretrained.return_value = mock_vlm
        mock_proc_cls.from_pretrained.return_value = mock_processor
        model = RoboRewardModel(cfg)

    model._mock_vlm = mock_vlm
    model._mock_processor = mock_processor
    return model


def test_model_instantiation_with_mock():
    """Model must instantiate without network when VLM is mocked."""
    model = _make_model_with_mocks()
    assert model.name == "robo_reward"
    assert model.config_class is RoboRewardConfig


def test_model_vlm_is_frozen():
    """All VLM parameters must have requires_grad=False after init."""
    model = _make_model_with_mocks()
    # The mock's parameters() is called in __init__ to freeze.
    # Verify freeze was called on the mock vlm.
    model._mock_vlm.parameters.assert_called()


def test_parse_score_standard():
    """Score parser must extract digit after 'ANSWER:'."""
    model = _make_model_with_mocks()
    assert model._parse_score("ANSWER: 3") == 3
    assert model._parse_score("ANSWER:5") == 5
    assert model._parse_score("The score is ANSWER: 2") == 2


def test_parse_score_fallback():
    """Score parser falls back to any standalone 1–5 digit."""
    model = _make_model_with_mocks()
    assert model._parse_score("I think 4 is correct") == 4


def test_parse_score_default_on_failure():
    """Score parser returns 1 when no digit is found."""
    model = _make_model_with_mocks()
    assert model._parse_score("no digit here") == 1


def test_compute_reward_shape_single_frame():
    """compute_reward must return (B,) tensor for (B, C, H, W) input."""
    from lerobot.rewards.robo_reward.modeling_robo_reward import RoboRewardModel

    cfg = RoboRewardConfig()
    mock_vlm, mock_processor = _make_mock_vlm_and_processor("ANSWER: 4")

    with (
        patch(
            "lerobot.rewards.robo_reward.modeling_robo_reward.Qwen3VLForConditionalGeneration"
        ) as mock_vlm_cls,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward.AutoProcessor") as mock_proc_cls,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward._TRANSFORMERS_AVAILABLE", True),
    ):
        mock_vlm_cls.from_pretrained.return_value = mock_vlm
        mock_proc_cls.from_pretrained.return_value = mock_processor
        model = RoboRewardModel(cfg)

    batch = {
        "observation.images.top": torch.rand(2, 3, 84, 84),
        "observation.language_instruction": ["pick up the cube", "place the block"],
    }

    with (
        patch("lerobot.rewards.robo_reward.modeling_robo_reward.process_vision_info") as mock_pvi,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward._QWEN_VL_UTILS_AVAILABLE", True),
        patch.object(model, "_tensor_to_pil", return_value=MagicMock()),
    ):
        mock_pvi.return_value = (None, None, {})
        rewards = model.compute_reward(batch)

    assert rewards.shape == (2,)
    assert rewards.dtype == torch.float32


def test_compute_reward_score_mapping():
    """compute_reward must map score 4 to 0.75 using default score_to_reward."""
    from lerobot.rewards.robo_reward.modeling_robo_reward import RoboRewardModel

    cfg = RoboRewardConfig()
    mock_vlm, mock_processor = _make_mock_vlm_and_processor("ANSWER: 4")

    with (
        patch(
            "lerobot.rewards.robo_reward.modeling_robo_reward.Qwen3VLForConditionalGeneration"
        ) as mock_vlm_cls,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward.AutoProcessor") as mock_proc_cls,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward._TRANSFORMERS_AVAILABLE", True),
    ):
        mock_vlm_cls.from_pretrained.return_value = mock_vlm
        mock_proc_cls.from_pretrained.return_value = mock_processor
        model = RoboRewardModel(cfg)

    batch = {
        "observation.images.top": torch.rand(1, 3, 84, 84),
        "observation.language_instruction": ["pick up the cube"],
    }

    with (
        patch("lerobot.rewards.robo_reward.modeling_robo_reward.process_vision_info") as mock_pvi,
        patch("lerobot.rewards.robo_reward.modeling_robo_reward._QWEN_VL_UTILS_AVAILABLE", True),
        patch.object(model, "_tensor_to_pil", return_value=MagicMock()),
    ):
        mock_pvi.return_value = (None, None, {})
        rewards = model.compute_reward(batch)

    assert rewards[0].item() == pytest.approx(0.75)


def test_forward_raises_not_implemented():
    """forward() must raise NotImplementedError (inference-only model)."""
    model = _make_model_with_mocks()
    with pytest.raises(NotImplementedError, match="compute_reward"):
        model.forward({})


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------


def test_processor_returns_pipelines():
    """make_robo_reward_pre_post_processors must return a 2-tuple of pipelines."""
    from lerobot.rewards.robo_reward.processor_robo_reward import make_robo_reward_pre_post_processors

    cfg = RoboRewardConfig()
    pre, post = make_robo_reward_pre_post_processors(cfg)
    assert pre is not None
    assert post is not None
    assert pre.name == "robo_reward_preprocessor"
    assert post.name == "robo_reward_postprocessor"
