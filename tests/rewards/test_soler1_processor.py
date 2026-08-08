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

"""Tests for SOLE-R1's stateful composite-image processor."""

from __future__ import annotations

import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.soler1.processor_soler1 import (
    SOLER1_COMPOSITE_KEY,
    SOLER1_IS_FIRST_KEY,
    SOLER1_TASK_KEY,
    SOLER1CompositeProcessorStep,
    _as_batched_chw_uint8,
)

EXTERNAL_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"


def _image(value: int) -> torch.Tensor:
    return torch.full((3, 4, 4), value, dtype=torch.uint8)


def _transition(
    external: torch.Tensor,
    *,
    wrist: torch.Tensor | None = None,
) -> dict:
    observation = {EXTERNAL_KEY: external}
    if wrist is not None:
        observation[WRIST_KEY] = wrist

    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.COMPLEMENTARY_DATA: {"task": "pick up the cube"},
    }


def test_as_batched_chw_uint8_converts_channel_last_float():
    image = torch.ones(2, 4, 4, 3)
    result = _as_batched_chw_uint8(image, name="image")

    assert result.shape == (2, 3, 4, 4)
    assert result.dtype == torch.uint8
    assert result.max().item() == 255


def test_external_composite_tracks_first_previous_current():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        composite_image_size=4,
        composite_padding=1,
    )

    first = step(_transition(_image(10)))
    first_composite = first[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_KEY]
    assert first_composite.shape == (1, 3, 4, 14)
    assert first[TransitionKey.OBSERVATION][SOLER1_IS_FIRST_KEY].tolist() == [True]

    second = step(_transition(_image(20)))
    second_composite = second[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_KEY]

    assert second[TransitionKey.OBSERVATION][SOLER1_IS_FIRST_KEY].tolist() == [False]
    assert torch.all(second_composite[..., :4] == 10)
    assert torch.all(second_composite[..., 5:9] == 10)
    assert torch.all(second_composite[..., 10:14] == 20)

    third = step(_transition(_image(30)))
    third_composite = third[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_KEY]

    assert torch.all(third_composite[..., :4] == 10)
    assert torch.all(third_composite[..., 5:9] == 20)
    assert torch.all(third_composite[..., 10:14] == 30)
    assert third[TransitionKey.OBSERVATION][SOLER1_TASK_KEY] == ["pick up the cube"]


def test_dual_view_composite_places_external_above_wrist():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
        composite_image_size=4,
        composite_padding=1,
    )

    output = step(_transition(_image(10), wrist=_image(100)))
    composite = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_KEY]

    assert composite.shape == (1, 3, 9, 14)
    assert torch.all(composite[..., :4, :4] == 10)
    assert torch.all(composite[..., 5:9, :4] == 100)


def test_from_zero_uses_only_first_and_current():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        from_zero=True,
        composite_image_size=4,
        composite_padding=1,
    )

    step(_transition(_image(10)))
    output = step(_transition(_image(30)))
    composite = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_KEY]

    assert composite.shape == (1, 3, 4, 9)
    assert torch.all(composite[..., :4] == 10)
    assert torch.all(composite[..., 5:9] == 30)


def test_reset_marks_next_observation_as_first():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        composite_image_size=4,
    )

    step(_transition(_image(10)))
    step(_transition(_image(20)))
    step.reset()

    output = step(_transition(_image(30)))
    assert output[TransitionKey.OBSERVATION][SOLER1_IS_FIRST_KEY].tolist() == [True]


def test_missing_configured_wrist_image_raises():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )

    with pytest.raises(KeyError, match="wrist image key"):
        step(_transition(_image(10)))


def test_batch_size_change_requires_reset():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        composite_image_size=4,
    )

    step(_transition(_image(10)))

    with pytest.raises(ValueError, match="batch size changed"):
        step(_transition(torch.stack([_image(20), _image(30)])))
