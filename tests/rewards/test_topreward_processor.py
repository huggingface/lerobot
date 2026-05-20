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

"""Tests for TOPReward's pre-processing helpers and encoder step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.rewards.topreward.processor_topreward import (
    TOPREWARD_FEATURE_PREFIX,
    _expand_tasks,
    _video_to_numpy,
)
from lerobot.types import TransitionKey
from tests.utils import skip_if_package_missing

# ---------------------------------------------------------------------------
# _video_to_numpy — pure (T, C, H, W) -> (T, H, W, C) uint8 conversion
# ---------------------------------------------------------------------------


def test_video_to_numpy_chw_float_is_converted_to_thwc_uint8():
    video = torch.rand(4, 3, 8, 8)
    array = _video_to_numpy(video, max_frames=None)

    assert array.shape == (4, 8, 8, 3)
    assert array.dtype == np.uint8
    assert array.min() >= 0 and array.max() <= 255


def test_video_to_numpy_already_thwc_uint8_passes_through():
    video = torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8)
    array = _video_to_numpy(video, max_frames=None)

    assert array.shape == (3, 8, 8, 3)
    assert array.dtype == np.uint8


def test_video_to_numpy_max_frames_tail_crops_recent_frames():
    video = torch.zeros(10, 3, 4, 4)
    for t in range(10):
        video[t] = t / 9.0

    array = _video_to_numpy(video, max_frames=3)

    assert array.shape == (3, 4, 4, 3)
    assert int(array[0, 0, 0, 0]) == int(round(7 / 9 * 255))
    assert int(array[-1, 0, 0, 0]) == 255


def test_video_to_numpy_rejects_3d_input():
    with pytest.raises(ValueError, match="Expected channel dim"):
        _video_to_numpy(torch.zeros(4, 8, 8), max_frames=None)


def test_video_to_numpy_floats_above_one_pass_through_without_rescaling():
    video = torch.full((1, 3, 2, 2), 5.0)
    array = _video_to_numpy(video, max_frames=None)

    assert array.shape == (1, 2, 2, 3)
    assert int(array.max()) == 5


def test_video_to_numpy_clips_very_large_floats_to_uint8_max():
    video = torch.full((1, 3, 2, 2), 300.0)
    array = _video_to_numpy(video, max_frames=None)

    assert int(array.max()) == 255


# ---------------------------------------------------------------------------
# _expand_tasks — string / list / tuple broadcasting to batch size
# ---------------------------------------------------------------------------


def test_expand_tasks_string_is_broadcast_to_batch_size():
    assert _expand_tasks("pick up", batch_size=3, default=None) == ["pick up", "pick up", "pick up"]


def test_expand_tasks_list_of_matching_size_passes_through():
    assert _expand_tasks(["a", "b", "c"], batch_size=3, default=None) == ["a", "b", "c"]


def test_expand_tasks_tuple_is_normalised_to_list():
    assert _expand_tasks(("a", "b"), batch_size=2, default=None) == ["a", "b"]


def test_expand_tasks_single_element_list_is_broadcast():
    assert _expand_tasks(["only one"], batch_size=3, default=None) == ["only one"] * 3


def test_expand_tasks_size_mismatch_raises():
    with pytest.raises(ValueError, match="Expected 3 tasks"):
        _expand_tasks(["a", "b"], batch_size=3, default=None)


def test_expand_tasks_missing_uses_default():
    assert _expand_tasks(None, batch_size=2, default="fallback") == ["fallback", "fallback"]


def test_expand_tasks_missing_without_default_raises():
    with pytest.raises(KeyError, match="task description"):
        _expand_tasks(None, batch_size=1, default=None)


def test_expand_tasks_wrong_type_raises():
    with pytest.raises(TypeError, match="must be a string or list"):
        _expand_tasks(42, batch_size=1, default=None)


# ---------------------------------------------------------------------------
# Encoder step — stubbed AutoProcessor + process_vision_info
# ---------------------------------------------------------------------------


def _skip_if_topreward_extras_missing(func):
    func = skip_if_package_missing("qwen-vl-utils", import_name="qwen_vl_utils")(func)
    func = skip_if_package_missing("transformers")(func)
    return func


class _FakeTokenizer:
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"

    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.zeros(1, 10, dtype=torch.long)}


class _FakeAutoProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()

    def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
        return "fake_prompt_text"

    def __call__(self, text=None, images=None, videos=None, **kwargs):  # noqa: ARG002
        seq_len = 10
        return {
            "input_ids": torch.randint(0, 100, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }


def _build_step(monkeypatch, **overrides):
    import importlib
    import sys
    import types

    from lerobot.rewards.topreward import processor_topreward
    from lerobot.utils import import_utils

    monkeypatch.setattr(processor_topreward, "AutoProcessor", _FakeAutoProcessor)

    # Stub qwen_vl_utils as a real module object (not MagicMock) so
    # ``require_package`` / ``find_spec`` don't choke on a missing ``__spec__``.
    fake_qwen_vl = types.ModuleType("qwen_vl_utils")
    fake_qwen_vl.process_vision_info = lambda messages: (None, None)  # type: ignore[attr-defined]
    fake_qwen_vl.__spec__ = importlib.machinery.ModuleSpec("qwen_vl_utils", None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl)

    # Clear the require_package cache so the stub is picked up.
    import_utils._require_package_cache.pop("qwen_vl_utils", None)

    return processor_topreward.TOPRewardEncoderProcessorStep(**overrides)


def _make_transition(observation: dict, complementary: dict | None = None) -> dict:
    transition: dict = {TransitionKey.OBSERVATION: observation}
    if complementary is not None:
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
    return transition


@_skip_if_topreward_extras_missing
def test_encoder_step_emits_input_ids_and_prompt_length(monkeypatch):
    """The processor must emit Qwen-VL tensors including ``input_ids`` and
    ``prompt_length`` under the ``observation.topreward.*`` namespace."""
    step = _build_step(monkeypatch)

    frames_batch = torch.zeros(1, 4, 3, 8, 8)
    out = step(
        _make_transition(
            observation={"observation.images.top": frames_batch},
            complementary={"task": "pick"},
        )
    )

    obs_out = out[TransitionKey.OBSERVATION]
    assert f"{TOPREWARD_FEATURE_PREFIX}input_ids" in obs_out
    assert f"{TOPREWARD_FEATURE_PREFIX}attention_mask" in obs_out
    assert f"{TOPREWARD_FEATURE_PREFIX}prompt_length" in obs_out

    prompt_length = obs_out[f"{TOPREWARD_FEATURE_PREFIX}prompt_length"]
    assert prompt_length.dtype == torch.long
    assert prompt_length.shape == (1,)


@_skip_if_topreward_extras_missing
def test_encoder_step_get_config_roundtrips_user_fields(monkeypatch):
    step = _build_step(
        monkeypatch,
        vlm_name="Qwen/Qwen3-VL-8B-Instruct",
        image_key="observation.images.cam_top",
        task_key="task",
        default_task="do the thing",
        max_frames=8,
        fps=4.0,
        add_chat_template=True,
        max_length=2048,
    )

    cfg = step.get_config()
    assert cfg["vlm_name"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg["image_key"] == "observation.images.cam_top"
    assert cfg["default_task"] == "do the thing"
    assert cfg["max_frames"] == 8
    assert cfg["fps"] == 4.0
    assert cfg["add_chat_template"] is True
    assert cfg["max_length"] == 2048


@_skip_if_topreward_extras_missing
def test_encoder_step_transform_features_is_identity(monkeypatch):
    step = _build_step(monkeypatch)
    features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.images.top": PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL),
        }
    }
    assert step.transform_features(features) == features


@_skip_if_topreward_extras_missing
def test_encoder_step_rejects_missing_image_key(monkeypatch):
    step = _build_step(monkeypatch, image_key="observation.images.top")
    with pytest.raises(KeyError, match="image key"):
        step(_make_transition(observation={}, complementary={"task": "pick"}))


@_skip_if_topreward_extras_missing
def test_encoder_step_rejects_non_dict_observation(monkeypatch):
    step = _build_step(monkeypatch)
    with pytest.raises(ValueError, match="observation dict"):
        step({TransitionKey.OBSERVATION: torch.zeros(1, 3, 8, 8)})
