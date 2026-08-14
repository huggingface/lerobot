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
from lerobot.rewards.factory import (
    get_reward_model_class,
    make_reward_model_config,
)
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.modeling_soler1 import _parse_progress
from lerobot.rewards.soler1.processor_soler1 import (
    COMPOSITE_WIDTH,
    SINGLE_VIEW_COMPOSITE_HEIGHT,
    SOLER1_COMPOSITE_IMAGE_KEY,
    SOLER1_ORIGINAL_LENGTH_KEY,
    SOLER1_SAMPLE_INDICES_KEY,
)
from tests.utils import skip_if_package_missing


class _FakeProcessor:
    def __init__(self) -> None:
        self.completion_batches: list[list[str]] = []
        self.user_prompts: list[str] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()

    def apply_chat_template(
        self,
        messages,
        **kwargs,  # noqa: ARG002
    ):
        self.user_prompts.append(messages[-1]["content"][-1]["text"])
        return "templated prompt"

    def __call__(
        self,
        text,
        images,
        **kwargs,  # noqa: ARG002
    ):
        batch_size = len(text)

        return {
            "input_ids": torch.ones(
                batch_size,
                8,
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                batch_size,
                8,
                dtype=torch.long,
            ),
            "pixel_values": torch.zeros(
                batch_size,
                3,
                4,
                4,
            ),
        }

    def batch_decode(
        self,
        completion_ids,
        **kwargs,  # noqa: ARG002
    ):
        if not self.completion_batches:
            raise AssertionError("No fake SOLE-R1 completion batch configured")

        completions = self.completion_batches.pop(0)

        assert len(completions) == completion_ids.shape[0]

        return completions


class _FakeQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()

    def generate(
        self,
        input_ids,
        **kwargs,  # noqa: ARG002
    ):
        suffix = torch.zeros(
            input_ids.shape[0],
            4,
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix], dim=-1)


def _patch_model(monkeypatch) -> None:
    from lerobot.rewards.soler1 import modeling_soler1

    monkeypatch.setattr(
        modeling_soler1,
        "AutoProcessor",
        _FakeProcessor,
    )
    monkeypatch.setattr(
        modeling_soler1,
        "AutoModelForImageTextToText",
        _FakeQwenModel,
    )


def _batch(
    *,
    batch_size: int = 1,
    trajectory_length: int = 1,
) -> dict[str, object]:
    composite_width = COMPOSITE_WIDTH

    return {
        SOLER1_COMPOSITE_IMAGE_KEY: torch.zeros(
            batch_size,
            trajectory_length,
            3,
            SINGLE_VIEW_COMPOSITE_HEIGHT,
            composite_width,
            dtype=torch.uint8,
        ),
        "task": ["pick up the cube"] * batch_size,
    }


def test_soler1_config_registered():
    assert "sole-r1" in RewardModelConfig.get_known_choices()
    assert RewardModelConfig.get_choice_class("sole-r1") is SOLER1Config
    assert isinstance(
        make_reward_model_config(
            "sole-r1",
            device="cpu",
        ),
        SOLER1Config,
    )


def test_soler1_factory_returns_in_tree_class():
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    assert get_reward_model_class("sole-r1") is SOLER1RewardModel


def test_soler1_config_validation():
    with pytest.raises(ValueError, match="reward_output"):
        SOLER1Config(
            device="cpu",
            reward_output="invalid",
        )

    with pytest.raises(ValueError, match="top_p"):
        SOLER1Config(
            device="cpu",
            top_p=0.0,
        )

    with pytest.raises(ValueError, match="min_progress"):
        SOLER1Config(
            device="cpu",
            min_progress=100.0,
            max_progress=0.0,
        )

    with pytest.raises(ValueError, match="reward_scale"):
        SOLER1Config(
            device="cpu",
            reward_scale=0.0,
        )

    with pytest.raises(ValueError, match="at least one"):
        SOLER1Config(
            device="cpu",
            external_image_key=None,
            wrist_image_key=None,
        )


def test_soler1_camera_features_are_channels_first():
    external_key = "observation.images.front"
    wrist_key = "observation.images.wrist"
    config = SOLER1Config(
        device="cpu",
        external_image_key=external_key,
        wrist_image_key=wrist_key,
    )
    assert config.input_features[external_key].shape == (3, 224, 224)
    assert config.input_features[wrist_key].shape == (3, 224, 224)
    assert config.output_features["reward"].shape == (1,)


def test_parse_progress():
    assert (
        _parse_progress(
            "<think>reasoning</think><answer>42%</answer>",
            minimum=-100.0,
            maximum=100.0,
        )
        == 42.0
    )

    assert (
        _parse_progress(
            "<answer>250%</answer>",
            minimum=-100.0,
            maximum=100.0,
        )
        == 100.0
    )

    assert (
        _parse_progress(
            "<answer>-250%</answer>",
            minimum=-100.0,
            maximum=100.0,
        )
        == -100.0
    )

    assert (
        _parse_progress(
            "reasoning says 10%, final estimate is 35%",
            minimum=-100.0,
            maximum=100.0,
        )
        == 35.0
    )

    assert (
        _parse_progress(
            "not parseable",
            minimum=-100.0,
            maximum=100.0,
        )
        is None
    )


@skip_if_package_missing("transformers")
def test_one_frame_trajectory_is_zero_without_generation(
    monkeypatch,
):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    reward = model.compute_reward(_batch(trajectory_length=1))

    assert reward.shape == (1,)
    assert reward.item() == 0.0
    assert model.last_completions == [""]
    assert model.last_reasoning_traces == [""]


@skip_if_package_missing("transformers")
def test_trajectory_uses_previous_prediction_and_returns_final(
    monkeypatch,
):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        ["<think>The gripper approached the cube.</think><answer>42%</answer>"],
        ["<think>The gripper grasped the cube.</think><answer>55%</answer>"],
    ]

    reward = model.compute_reward(_batch(trajectory_length=3))

    assert reward.shape == (1,)
    assert reward.item() == pytest.approx(0.55)

    assert "previous timestep is 42%" in model.processor.user_prompts[-1]
    assert model.last_completions == ["<think>The gripper grasped the cube.</think><answer>55%</answer>"]
    assert model.last_reasoning_traces == ["The gripper grasped the cube."]


@skip_if_package_missing("transformers")
def test_returns_one_sparse_value_per_trajectory(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        [
            "<answer>25%</answer>",
            "<answer>75%</answer>",
        ]
    ]

    rewards = model.compute_reward(
        _batch(
            batch_size=2,
            trajectory_length=2,
        )
    )

    assert rewards.shape == (2,)
    assert rewards.tolist() == pytest.approx([0.25, 0.75])


@skip_if_package_missing("transformers")
def test_dense_progress_returns_one_value_per_timestep(
    monkeypatch,
):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        ["<answer>42%</answer>"],
        ["<answer>55%</answer>"],
    ]

    rewards = model.compute_reward(
        _batch(trajectory_length=3),
        dense=True,
    )

    assert rewards.shape == (1, 3)
    assert rewards.tolist()[0] == pytest.approx([0.0, 0.42, 0.55])


@skip_if_package_missing("transformers")
def test_dense_progress_supports_multiple_trajectories(
    monkeypatch,
):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        [
            "<answer>20%</answer>",
            "<answer>60%</answer>",
        ],
        [
            "<answer>40%</answer>",
            "<answer>80%</answer>",
        ],
    ]

    rewards = model.compute_reward(
        _batch(
            batch_size=2,
            trajectory_length=3,
        ),
        dense=True,
    )

    assert rewards.shape == (2, 3)
    assert rewards.tolist()[0] == pytest.approx([0.0, 0.20, 0.40])
    assert rewards.tolist()[1] == pytest.approx([0.0, 0.60, 0.80])


@skip_if_package_missing("transformers")
def test_dense_false_returns_only_final_progress(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        ["<answer>42%</answer>"],
        ["<answer>55%</answer>"],
    ]

    reward = model.compute_reward(
        _batch(trajectory_length=3),
        dense=False,
    )

    assert reward.shape == (1,)
    assert reward.item() == pytest.approx(0.55)


@skip_if_package_missing("transformers")
def test_dense_success_checks_only_final_progress(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            reward_output="success",
            success_threshold=0.5,
        )
    )

    model.processor.completion_batches = [
        [
            "<answer>90%</answer>",
            "<answer>10%</answer>",
        ],
        [
            "<answer>40%</answer>",
            "<answer>60%</answer>",
        ],
    ]

    rewards = model.compute_reward(
        _batch(
            batch_size=2,
            trajectory_length=3,
        ),
        dense=True,
    )

    # Final progress values are 0.40 and 0.60. The intermediate
    # 0.90 value must not make the first trajectory successful.
    assert rewards.shape == (2,)
    assert rewards.tolist() == [0.0, 1.0]


@skip_if_package_missing("transformers")
def test_sparse_success_checks_final_progress(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            reward_output="success",
            success_threshold=0.5,
        )
    )

    model.processor.completion_batches = [
        [
            "<answer>49%</answer>",
            "<answer>51%</answer>",
        ]
    ]

    rewards = model.compute_reward(
        _batch(
            batch_size=2,
            trajectory_length=2,
        )
    )

    assert rewards.shape == (2,)
    assert rewards.tolist() == [0.0, 1.0]


@skip_if_package_missing("transformers")
def test_invalid_completion_falls_back_to_previous(
    monkeypatch,
):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        ["<answer>42%</answer>"],
        ["invalid output"],
    ]

    rewards = model.compute_reward(
        _batch(trajectory_length=3),
        dense=True,
    )

    assert rewards.shape == (1, 3)
    assert rewards.tolist()[0] == pytest.approx([0.0, 0.42, 0.42])


@skip_if_package_missing("transformers")
def test_invalid_completion_can_raise(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            fallback_to_previous=False,
        )
    )

    model.processor.completion_batches = [
        ["invalid output"],
    ]

    with pytest.raises(
        ValueError,
        match="Could not parse SOLE-R1 completion",
    ):
        model.compute_reward(_batch(trajectory_length=2))


@skip_if_package_missing("transformers")
def test_input_length_limit_is_checked(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            max_input_length=4,
        )
    )

    with pytest.raises(
        ValueError,
        match="input length",
    ):
        model.compute_reward(_batch(trajectory_length=2))


@skip_if_package_missing("transformers")
def test_reset_clears_diagnostic_state(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        ["<think>reasoning</think><answer>42%</answer>"],
    ]

    model.compute_reward(_batch(trajectory_length=2))

    assert model._previous_progress == [42.0]
    assert model.last_completions
    assert model.last_reasoning_traces == ["reasoning"]

    model.reset()

    assert model._previous_progress is None
    assert model.last_completions == []
    assert model.last_reasoning_traces == []


@skip_if_package_missing("transformers")
def test_soler1_save_load_is_config_only(
    monkeypatch,
    tmp_path,
):
    from huggingface_hub.constants import (
        CONFIG_NAME,
        SAFETENSORS_SINGLE_FILE,
    )

    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)

    config = SOLER1Config(
        device="cpu",
        external_image_key="observation.images.front",
        wrist_image_key="observation.images.wrist",
        reward_output="success",
        success_threshold=0.75,
        reward_scale=0.01,
    )
    model = SOLER1RewardModel(config)

    model.save_pretrained(tmp_path)

    assert (tmp_path / CONFIG_NAME).exists()
    assert not (tmp_path / SAFETENSORS_SINGLE_FILE).exists()

    reloaded = SOLER1RewardModel.from_pretrained(tmp_path)

    assert isinstance(reloaded.config, SOLER1Config)
    assert reloaded.config.external_image_key == "observation.images.front"
    assert reloaded.config.wrist_image_key == "observation.images.wrist"
    assert reloaded.config.reward_output == "success"
    assert reloaded.config.success_threshold == 0.75
    assert reloaded.config.reward_scale == 0.01


@skip_if_package_missing("transformers")
def test_soler1_is_not_trainable(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    assert model.is_trainable is False

    with pytest.raises(
        NotImplementedError,
        match="not trainable",
    ):
        model.forward({"x": torch.zeros(1)})


@skip_if_package_missing("transformers")
def test_unbatched_trajectory_returns_unbatched_rewards(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))
    model.processor.completion_batches = [
        ["<answer>20%</answer>"],
        ["<answer>50%</answer>"],
    ]

    batch = _batch(trajectory_length=3)
    batch[SOLER1_COMPOSITE_IMAGE_KEY] = batch[SOLER1_COMPOSITE_IMAGE_KEY].squeeze(0)
    batch["task"] = "pick up the cube"

    dense = model.compute_reward(batch, dense=True)

    assert dense.shape == (3,)
    assert dense.tolist() == pytest.approx([0.0, 0.2, 0.5])


@skip_if_package_missing("transformers")
def test_dense_progress_interpolates_sampled_predictions(
    monkeypatch,
):
    from lerobot.rewards.soler1.modeling_soler1 import (
        SOLER1RewardModel,
    )

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(SOLER1Config(device="cpu"))

    model.processor.completion_batches = [
        ["<answer>20%</answer>"],
        ["<answer>60%</answer>"],
    ]

    # Three composites correspond to original indices [0, 2, 4].
    batch = _batch(trajectory_length=3)
    batch[SOLER1_SAMPLE_INDICES_KEY] = torch.tensor([0, 2, 4])
    batch[SOLER1_ORIGINAL_LENGTH_KEY] = torch.tensor(5)

    rewards = model.compute_reward(
        batch,
        dense=True,
    )

    assert rewards.shape == (1, 5)
    assert rewards.tolist()[0] == pytest.approx([0.0, 0.1, 0.2, 0.4, 0.6])


@skip_if_package_missing("transformers")
def test_wrist_only_uses_wrist_prompt(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            external_image_key=None,
            wrist_image_key="observation.images.wrist",
        )
    )
    model.processor.completion_batches = [
        ["<answer>10%</answer>"],
    ]

    model.compute_reward(_batch(trajectory_length=2))

    assert "robot's wrist camera" in model.processor.user_prompts[0]


@skip_if_package_missing("transformers")
def test_missing_task_raises_without_default(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            default_task=None,
        )
    )

    batch = _batch(trajectory_length=2)
    del batch["task"]

    with pytest.raises(KeyError, match="task description"):
        model.compute_reward(batch)


@skip_if_package_missing("transformers")
def test_explicit_default_task_is_supported(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel

    _patch_model(monkeypatch)
    model = SOLER1RewardModel(
        SOLER1Config(
            device="cpu",
            default_task="pick up the cube",
        )
    )
    model.processor.completion_batches = [["<think>closer</think><answer>25%</answer>"]]

    batch = _batch(trajectory_length=2)
    del batch["task"]

    reward = model.compute_reward(batch)

    torch.testing.assert_close(reward, torch.tensor([0.25]))


@skip_if_package_missing("transformers")
def test_unbatched_tchw_input_produces_one_trajectory_reward(monkeypatch):
    from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel
    from lerobot.rewards.soler1.processor_soler1 import (
        make_soler1_pre_post_processors,
    )

    _patch_model(monkeypatch)

    config = SOLER1Config(
        device="cpu",
        external_image_key="observation.images.front",
        wrist_image_key=None,
        downsample_to=None,
    )
    model = SOLER1RewardModel(config)
    preprocessor, _ = make_soler1_pre_post_processors(config)

    trajectory = torch.randint(
        0,
        256,
        (3, 3, 32, 32),
        dtype=torch.uint8,
    )

    batch = preprocessor(
        {
            config.external_image_key: trajectory,
            config.task_key: "pick up the cube",
        }
    )

    model.processor.completion_batches = [
        ["<think>some progress</think><answer>25%</answer>"],
        ["<think>more progress</think><answer>50%</answer>"],
    ]

    reward = model.compute_reward(batch)

    assert reward.shape == (1,)
    torch.testing.assert_close(reward, torch.tensor([0.5]))

    # Restore the mocked completions because the first call consumed them.
    model.processor.completion_batches = [
        ["<think>some progress</think><answer>25%</answer>"],
        ["<think>more progress</think><answer>50%</answer>"],
    ]

    dense_reward = model.compute_reward(batch, dense=True)

    assert dense_reward.shape == (1, 3)
    torch.testing.assert_close(
        dense_reward,
        torch.tensor([[0.0, 0.25, 0.5]]),
    )
