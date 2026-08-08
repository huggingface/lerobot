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

"""Unit tests for the inference-only SOLE-R1 reward model."""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.modeling_soler1 import _parse_progress
from lerobot.rewards.soler1.processor_soler1 import (
    SOLER1_COMPOSITE_KEY,
    SOLER1_IS_FIRST_KEY,
    SOLER1_TASK_KEY,
)
from tests.utils import skip_if_package_missing


class _FakeProcessor:
    def __init__(self) -> None:
        self.completions = ["<think>progress</think><answer>42%</answer>"]
        self.user_prompts: list[str] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()

    def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
        self.user_prompts.append(messages[-1]["content"][-1]["text"])
        return "templated prompt"

    def __call__(self, text, images, **kwargs):  # noqa: ARG002
        batch_size = len(text)
        return {
            "input_ids": torch.ones(batch_size, 8, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size, 3, 4, 4),
        }

    def batch_decode(self, completion_ids, **kwargs):  # noqa: ARG002
        return self.completions[: completion_ids.shape[0]]


class _FakeQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()

    def generate(self, input_ids, **kwargs):  # noqa: ARG002
        suffix = torch.zeros(input_ids.shape[0], 4, dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=-1)


def _patch_model(monkeypatch) -> None:
    from lerobot.rewards.soler1 import modeling_soler1

    monkeypatch.setattr(modeling_soler1, "AutoProcessor", _FakeProcessor)
    monkeypatch.setattr(
        modeling_soler1,
        "Qwen3VLForConditionalGeneration",
        _FakeQwenModel,
    )


def _batch(*, is_first: bool) -> dict[str, object]:
    return {
        SOLER1_COMPOSITE_KEY: torch.zeros(1, 3, 8, 24, dtype=torch.uint8),
        SOLER1_IS_FIRST_KEY: torch.tensor([is_first]),
        SOLER1_TASK_KEY: ["pick up the cube"],
    }


def test_soler1_config_registered():
    assert "soler1" in RewardModelConfig.get_known_choices()
    assert RewardModelConfig.get_choice_class("soler1") is SOLER1Config
    assert isinstance(make_reward_model_config("soler1", device="cpu"), SOLER1Config)


def test_soler1_factory_returns_in_tree_class():
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    assert get_reward_model_class("soler1") is SOLER1RewardModel


def test_soler1_config_validation():
    with pytest.raises(ValueError, match="composite_image_size"):
        SOLER1Config(device="cpu", composite_image_size=0)
    with pytest.raises(ValueError, match="top_p"):
        SOLER1Config(device="cpu", top_p=0.0)
    with pytest.raises(ValueError, match="min_progress"):
        SOLER1Config(device="cpu", min_progress=100, max_progress=0)


def test_parse_progress():
    assert (
        _parse_progress(
            "<think>reasoning</think><answer>42%</answer>",
            minimum=-100,
            maximum=100,
        )
        == 42
    )
    assert (
        _parse_progress(
            "<answer>250%</answer>",
            minimum=-100,
            maximum=100,
        )
        == 100
    )
    assert _parse_progress("not parseable", minimum=-100, maximum=100) is None


@skip_if_package_missing("transformers")
def test_first_timestep_is_zero_without_generation(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    reward = model.compute_reward(_batch(is_first=True))

    assert reward.shape == (1,)
    assert reward.item() == 0.0
    assert model.last_reasoning_traces == [] or model.last_reasoning_traces == [""]


@skip_if_package_missing("transformers")
def test_soler1_uses_previous_prediction_in_next_prompt(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.compute_reward(_batch(is_first=True))

    model.processor.completions = ["<think>The gripper approached the cube.</think><answer>42%</answer>"]
    second_reward = model.compute_reward(_batch(is_first=False))
    assert second_reward.item() == pytest.approx(0.42)

    model.processor.completions = ["<think>The gripper grasped the cube.</think><answer>55%</answer>"]
    third_reward = model.compute_reward(_batch(is_first=False))

    assert third_reward.item() == pytest.approx(0.55)
    assert "previous timestep is 42%" in model.processor.user_prompts[-1]
    assert "<answer>55%</answer>" in model.last_reasoning_traces[0]


@skip_if_package_missing("transformers")
def test_invalid_completion_falls_back_to_previous(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.compute_reward(_batch(is_first=True))
    model.processor.completions = ["<answer>42%</answer>"]
    model.compute_reward(_batch(is_first=False))

    model.processor.completions = ["invalid output"]
    reward = model.compute_reward(_batch(is_first=False))

    assert reward.item() == pytest.approx(0.42)


@skip_if_package_missing("transformers")
def test_from_zero_prompt_omits_previous_progress(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu", from_zero=True))

    model.compute_reward(_batch(is_first=True))
    model.compute_reward(_batch(is_first=False))

    assert "previous timestep" not in model.processor.user_prompts[-1]


@skip_if_package_missing("transformers")
def test_reset_restarts_episode_at_zero(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.compute_reward(_batch(is_first=True))
    model.compute_reward(_batch(is_first=False))
    model.reset()

    reward = model.compute_reward(_batch(is_first=True))
    assert reward.item() == 0.0


@skip_if_package_missing("transformers")
def test_soler1_save_load_is_config_only(monkeypatch, tmp_path):
    from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    config = SOLER1Config(
        device="cpu",
        external_image_key="observation.images.front",
        wrist_image_key="observation.images.wrist",
        reward_scale=1.0,
    )
    model = SOLER1RewardModel(config)
    model.save_pretrained(tmp_path)

    assert (tmp_path / CONFIG_NAME).exists()
    assert not (tmp_path / SAFETENSORS_SINGLE_FILE).exists()

    reloaded = SOLER1RewardModel.from_pretrained(tmp_path)
    assert isinstance(reloaded.config, SOLER1Config)
    assert reloaded.config.external_image_key == "observation.images.front"
    assert reloaded.config.wrist_image_key == "observation.images.wrist"
    assert reloaded.config.reward_scale == 1.0


@skip_if_package_missing("transformers")
def test_soler1_is_not_trainable(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    assert model.is_trainable is False
    with pytest.raises(NotImplementedError, match="not trainable"):
        model.forward({"x": torch.zeros(1)})
