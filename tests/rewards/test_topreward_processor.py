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
    TOPRewardEncoderProcessorStep,
    _expand_tasks,
    _video_to_numpy,
)
from lerobot.types import TransitionKey

# ---------------------------------------------------------------------------
# _video_to_numpy — pure (T, C, H, W) -> (T, H, W, C) uint8 conversion
# ---------------------------------------------------------------------------


def test_video_to_numpy_chw_float_is_converted_to_thwc_uint8():
    video = torch.rand(4, 3, 8, 8)  # (T, C, H, W) floats in [0, 1]
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
    """``max_frames`` should keep the **last** K frames (most recent)."""
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
    """If ``array.max() > 1`` the helper assumes the tensor is already in the
    uint8 range; values pass through unchanged (but are still clipped to 255)."""
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
# Encoder step — input/output shapes + dataclass surface
# ---------------------------------------------------------------------------


def _make_transition(observation: dict, complementary: dict | None = None) -> dict:
    """Build a tiny ``EnvTransition`` dict for the encoder step."""
    transition: dict = {TransitionKey.OBSERVATION: observation}
    if complementary is not None:
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
    return transition


def test_encoder_step_writes_namespaced_frames_and_task():
    """The encoder step's output is the contract the model reads from. It
    must populate exactly two namespaced keys: ``frames`` and ``task``."""
    step = TOPRewardEncoderProcessorStep(
        image_key="observation.images.top",
        task_key="task",
        max_frames=None,
    )

    frames_batch = torch.zeros(2, 4, 3, 8, 8)  # (B=2, T=4, C, H, W)
    out = step(
        _make_transition(
            observation={"observation.images.top": frames_batch},
            complementary={"task": ["pick", "place"]},
        )
    )

    obs_out = out[TransitionKey.OBSERVATION]
    frames_out = obs_out[f"{TOPREWARD_FEATURE_PREFIX}frames"]
    tasks_out = obs_out[f"{TOPREWARD_FEATURE_PREFIX}task"]

    assert len(frames_out) == 2
    assert all(arr.shape == (4, 8, 8, 3) and arr.dtype == np.uint8 for arr in frames_out)
    assert tasks_out == ["pick", "place"]


def test_encoder_step_adds_singleton_time_dim_for_4d_input():
    """A ``(B, C, H, W)`` observation is the single-frame case; the encoder
    must unsqueeze the time dim so the model still sees a video."""
    step = TOPRewardEncoderProcessorStep(image_key="observation.images.top", max_frames=None)

    frames_batch = torch.zeros(1, 3, 8, 8)  # (B=1, C, H, W) — no time dim
    out = step(
        _make_transition(
            observation={"observation.images.top": frames_batch},
            complementary={"task": "pick"},
        )
    )

    frames_out = out[TransitionKey.OBSERVATION][f"{TOPREWARD_FEATURE_PREFIX}frames"]
    assert len(frames_out) == 1
    assert frames_out[0].shape == (1, 8, 8, 3)  # (T=1, H, W, C)


def test_encoder_step_uses_default_task_when_complementary_is_missing():
    step = TOPRewardEncoderProcessorStep(
        image_key="observation.images.top",
        default_task="perform the task",
    )

    frames_batch = torch.zeros(1, 2, 3, 4, 4)
    out = step(_make_transition(observation={"observation.images.top": frames_batch}))

    tasks_out = out[TransitionKey.OBSERVATION][f"{TOPREWARD_FEATURE_PREFIX}task"]
    assert tasks_out == ["perform the task"]


def test_encoder_step_rejects_missing_image_key():
    step = TOPRewardEncoderProcessorStep(image_key="observation.images.top")
    with pytest.raises(KeyError, match="image key"):
        step(_make_transition(observation={}, complementary={"task": "pick"}))


def test_encoder_step_rejects_non_dict_observation():
    step = TOPRewardEncoderProcessorStep()
    with pytest.raises(ValueError, match="observation dict"):
        step({TransitionKey.OBSERVATION: torch.zeros(1, 3, 8, 8)})


def test_encoder_step_rejects_3d_or_6d_input():
    """The encoder accepts ``(B,C,H,W)`` or ``(B,T,C,H,W)`` only."""
    step = TOPRewardEncoderProcessorStep(image_key="observation.images.top")
    with pytest.raises(ValueError, match=r"\(B,C,H,W\)"):
        step(
            _make_transition(
                observation={"observation.images.top": torch.zeros(8, 8, 3)},
                complementary={"task": "pick"},
            )
        )


def test_encoder_step_get_config_roundtrips_user_fields():
    """``get_config`` must serialise every user-tunable field — these are
    what the processor pipeline saves under ``preprocessor_config.json``."""
    step = TOPRewardEncoderProcessorStep(
        image_key="observation.images.cam_top",
        task_key="task",
        default_task="do the thing",
        max_frames=8,
    )

    assert step.get_config() == {
        "image_key": "observation.images.cam_top",
        "task_key": "task",
        "default_task": "do the thing",
        "max_frames": 8,
    }


def test_encoder_step_transform_features_is_identity():
    """The encoder writes plain Python objects (numpy arrays / strings)
    into ``observation`` at call time but does NOT advertise new typed
    features at pipeline-build time — the model reads them via the
    ``TOPREWARD_FEATURE_PREFIX`` namespace, not via the typed feature map.
    """
    step = TOPRewardEncoderProcessorStep()
    features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.images.top": PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL),
        }
    }
    assert step.transform_features(features) == features
