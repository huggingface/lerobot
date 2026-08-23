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

from typing import Any

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.modeling_soler1 import _parse_progress, extract_reasoning_trace
from lerobot.rewards.soler1.processor_soler1 import (
    SOLER1_IMAGE_GRID_THW_KEY,
    SOLER1_IMAGE_TOKEN_COUNT_KEY,
    SOLER1_PIXEL_VALUES_KEY,
)
from tests.utils import skip_if_package_missing

EXTERNAL_KEY = "observation.images.external"
WRIST_KEY = "observation.images.wrist"


class _FakeTokenizer:
    image_token = "<|image_pad|>"
    image_token_id = 1
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self.completion_batches: list[list[str]] = []
        self.user_prompts: list[str] = []
        self.tokenized_texts: list[list[str]] = []

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeTokenizer:  # noqa: ARG003
        return cls()

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:  # noqa: ARG002
        user_prompt = messages[-1]["content"][-1]["text"]
        self.user_prompts.append(user_prompt)
        return f"{self.image_token}{user_prompt}"

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:  # noqa: ARG002
        self.tokenized_texts.append(list(texts))
        batch_size = len(texts)
        return {
            "input_ids": torch.ones(batch_size, 8, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
        }

    def batch_decode(self, completion_ids: torch.Tensor, **kwargs: Any) -> list[str]:  # noqa: ARG002
        if not self.completion_batches:
            raise AssertionError("No fake SOLE-R1 completion batch configured")
        completions = self.completion_batches.pop(0)
        assert len(completions) == completion_ids.shape[0]
        return completions


class _FakeQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeQwenModel:  # noqa: ARG003
        return cls()

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        suffix = torch.zeros(
            input_ids.shape[0],
            4,
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix], dim=-1)


def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1 import modeling_soler1

    monkeypatch.setattr(modeling_soler1, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(modeling_soler1, "AutoModelForImageTextToText", _FakeQwenModel)


def _prepared_batch(
    *,
    batch_size: int = 1,
    trajectory_length: int = 1,
    tasks: list[str] | None = None,
) -> dict[str, object]:
    if tasks is None:
        tasks = ["pick up the cube"] * batch_size

    return {
        SOLER1_PIXEL_VALUES_KEY: torch.zeros(batch_size, trajectory_length, 4, 8),
        SOLER1_IMAGE_GRID_THW_KEY: torch.tensor([1, 2, 2]).repeat(batch_size, trajectory_length, 1),
        SOLER1_IMAGE_TOKEN_COUNT_KEY: torch.ones(batch_size, trajectory_length, dtype=torch.long),
        "task": tasks,
    }


def test_soler1_registered_under_hyphenated_user_argument() -> None:
    assert RewardModelConfig.get_choice_class("sole-r1") is SOLER1Config
    assert isinstance(make_reward_model_config("sole-r1", device="cpu"), SOLER1Config)

    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    assert get_reward_model_class("sole-r1") is SOLER1RewardModel


def test_config_matches_public_generation_defaults() -> None:
    config = SOLER1Config(device="cpu")

    assert config.max_new_tokens == 200
    assert config.temperature == 1.0
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.input_features == {}
    assert config.output_features["reward"].shape == (1,)


@pytest.mark.parametrize(
    ("external_key", "wrist_key"),
    [
        (EXTERNAL_KEY, None),
        (None, WRIST_KEY),
        (EXTERNAL_KEY, WRIST_KEY),
    ],
)
def test_config_supports_all_camera_modes(
    external_key: str | None,
    wrist_key: str | None,
) -> None:
    config = SOLER1Config(
        device="cpu",
        external_image_key=external_key,
        wrist_image_key=wrist_key,
    )

    assert config.external_image_key == external_key
    assert config.wrist_image_key == wrist_key


def test_config_rejects_no_camera() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SOLER1Config(
            device="cpu",
            external_image_key=None,
            wrist_image_key=None,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_k": -1}, "top_k"),
        ({"reward_scale": 0.0}, "reward_scale"),
        ({"reward_output": "invalid"}, "reward_output"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SOLER1Config(device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<think>x</think><answer>42%</answer>", 42.0),
        ("<answer>12.5</answer>", 12.5),
        ("<answer>250%</answer>", 100.0),
        ("<answer>-250%</answer>", -100.0),
        ("<answer>10%</answer><answer>35%</answer>", 35.0),
        ("untagged 35%", None),
        ("not parseable", None),
    ],
)
def test_parse_progress(completion: str, expected: float | None) -> None:
    assert _parse_progress(completion, minimum=-100.0, maximum=100.0) == expected


def test_extract_reasoning_trace() -> None:
    assert extract_reasoning_trace("<think>careful reasoning</think><answer>10%</answer>") == (
        "careful reasoning"
    )
    assert extract_reasoning_trace("<answer>10%</answer>") == ""


@skip_if_package_missing("transformers")
@pytest.mark.parametrize(
    ("external_key", "wrist_key", "expected_text"),
    [
        (EXTERNAL_KEY, None, "shown to the left"),
        (None, WRIST_KEY, "robot's wrist camera"),
        (EXTERNAL_KEY, WRIST_KEY, "views on the top are from an external camera"),
    ],
)
def test_prompt_matches_camera_configuration(
    monkeypatch: pytest.MonkeyPatch,
    external_key: str | None,
    wrist_key: str | None,
    expected_text: str,
) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            external_image_key=external_key,
            wrist_image_key=wrist_key,
        )
    )

    prompt = model._build_prompt(
        task_description="pick up the cube",
        previous_progress=25.0,
    )

    assert expected_text in prompt
    assert "previous timestep is 25%" in prompt


@skip_if_package_missing("transformers")
def test_one_timestep_is_zero_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    reward = model.compute_reward(_prepared_batch())

    assert reward.shape == (1,)
    assert reward.item() == 0.0
    assert model.model.generate_calls == []
    assert model.last_completions == [[""]]
    assert model.last_reasoning_traces == [[""]]


@skip_if_package_missing("transformers")
def test_multiple_timesteps_and_batches_return_b_rewards_and_full_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))
    model.tokenizer.completion_batches = [
        [
            "<think>a1</think><answer>20%</answer>",
            "<think>b1</think><answer>60%</answer>",
        ],
        [
            "<think>a2</think><answer>40%</answer>",
            "<think>b2</think><answer>80%</answer>",
        ],
    ]

    rewards = model.compute_reward(
        _prepared_batch(
            batch_size=2,
            trajectory_length=3,
            tasks=["pick up the cube", "open the drawer"],
        )
    )

    assert rewards.shape == (2,)
    assert rewards.tolist() == pytest.approx([0.4, 0.8])
    assert model.last_reasoning_traces == [["", "a1", "a2"], ["", "b1", "b2"]]
    assert model.last_completions[0][1].endswith("<answer>20%</answer>")
    assert model.last_completions[1][2].endswith("<answer>80%</answer>")
    assert "previous timestep is 20%" in model.tokenizer.user_prompts[-2]
    assert "previous timestep is 60%" in model.tokenizer.user_prompts[-1]


@skip_if_package_missing("transformers")
def test_dense_progress_is_bt_and_uses_prepared_vision_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))
    model.tokenizer.completion_batches = [["<answer>25%</answer>", "<answer>75%</answer>"]]

    progress = model.compute_progress(_prepared_batch(batch_size=2, trajectory_length=2))

    assert progress.shape == (2, 2)
    torch.testing.assert_close(
        progress,
        torch.tensor([[0.0, 0.25], [0.0, 0.75]]),
    )

    call = model.model.generate_calls[0]
    assert call["pixel_values"].shape == (8, 8)
    assert call["image_grid_thw"].shape == (2, 3)
    assert call["mm_token_type_ids"].shape == (2, 8)


@skip_if_package_missing("transformers")
def test_invalid_completion_falls_back_to_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))
    model.tokenizer.completion_batches = [
        ["<answer>42%</answer>"],
        ["invalid"],
    ]

    progress = model.compute_progress(_prepared_batch(trajectory_length=3))

    torch.testing.assert_close(progress, torch.tensor([[0.0, 0.42, 0.42]]))


@skip_if_package_missing("transformers")
def test_invalid_completion_raises_when_fallback_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            fallback_to_previous=False,
        )
    )
    model.tokenizer.completion_batches = [["invalid"]]

    with pytest.raises(ValueError, match="Could not parse SOLE-R1 completion"):
        model.compute_reward(_prepared_batch(trajectory_length=2))


@skip_if_package_missing("transformers")
def test_success_output_uses_only_final_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            reward_output="success",
            success_threshold=0.5,
        )
    )
    model.tokenizer.completion_batches = [
        ["<answer>90%</answer>", "<answer>10%</answer>"],
        ["<answer>40%</answer>", "<answer>80%</answer>"],
    ]

    reward = model.compute_reward(_prepared_batch(batch_size=2, trajectory_length=3))

    assert reward.shape == (2,)
    assert reward.tolist() == [0.0, 1.0]


@skip_if_package_missing("transformers")
def test_greedy_generation_omits_sampling_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu", temperature=0.0))

    kwargs = model._generation_kwargs()

    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 200
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


@skip_if_package_missing("transformers")
def test_model_rejects_unprepared_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    with pytest.raises(KeyError, match="statically prepared"):
        model.compute_reward({"task": "pick up the cube"})


@skip_if_package_missing("transformers")
def test_model_rejects_inconsistent_prepared_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))
    batch = _prepared_batch(batch_size=2, trajectory_length=2)
    batch[SOLER1_IMAGE_TOKEN_COUNT_KEY] = torch.ones(1, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="disagree on batch/time dimensions"):
        model.compute_reward(batch)
