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

"""Tests for SOLE-R1's trajectory composite-image processor."""

from __future__ import annotations

import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.soler1.processor_soler1 import (
    COMPOSITE_WIDTH,
    FRAME_SIZE,
    PADDING,
    SOLER1_COMPOSITE_IMAGE_KEY,
    SOLER1_ORIGINAL_LENGTH_KEY,
    SOLER1_SAMPLE_INDICES_KEY,
    SOLER1CompositeProcessorStep,
    _to_btchw_uint8,
)

EXTERNAL_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"


def _image(value: int) -> torch.Tensor:
    return torch.full(
        (3, 4, 4),
        value,
        dtype=torch.uint8,
    )


def _video(values: list[int]) -> torch.Tensor:
    return torch.stack(
        [_image(value) for value in values],
        dim=0,
    )


def _batched_video(
    trajectories: list[list[int]],
) -> torch.Tensor:
    return torch.stack(
        [_video(values) for values in trajectories],
        dim=0,
    )


def _transition(
    external: torch.Tensor,
    *,
    wrist: torch.Tensor | None = None,
) -> dict:
    observation = {
        EXTERNAL_KEY: external,
    }

    if wrist is not None:
        observation[WRIST_KEY] = wrist

    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "pick up the cube",
        },
    }


def test_to_btchw_uint8_adds_batch_dimension_to_tchw():
    images = torch.stack(
        [
            torch.full((3, 4, 4), 10, dtype=torch.uint8),
            torch.full((3, 4, 4), 20, dtype=torch.uint8),
        ]
    )

    result = _to_btchw_uint8(
        images,
        image_key="image",
    )

    assert result.shape == (1, 2, 3, 4, 4)
    assert result.dtype == torch.uint8
    assert torch.all(result[:, 0] == 10)
    assert torch.all(result[:, 1] == 20)


def test_to_btchw_uint8_preserves_btchw():
    images = torch.ones(
        2,
        5,
        3,
        4,
        4,
        dtype=torch.uint8,
    )

    result = _to_btchw_uint8(
        images,
        image_key="image",
    )

    assert result.shape == (2, 5, 3, 4, 4)
    assert result.dtype == torch.uint8


def test_to_btchw_uint8_converts_channel_last_float():
    images = torch.ones(
        2,
        5,
        4,
        4,
        3,
    )

    result = _to_btchw_uint8(
        images,
        image_key="image",
    )

    assert result.shape == (2, 5, 3, 4, 4)
    assert result.dtype == torch.uint8
    assert result.max().item() == 255


def test_to_btchw_uint8_expands_grayscale():
    images = torch.ones(
        2,
        5,
        1,
        4,
        4,
        dtype=torch.uint8,
    )

    result = _to_btchw_uint8(
        images,
        image_key="image",
    )

    assert result.shape == (2, 5, 3, 4, 4)
    assert torch.equal(
        result[:, :, 0],
        result[:, :, 1],
    )
    assert torch.equal(
        result[:, :, 1],
        result[:, :, 2],
    )


def test_to_btchw_uint8_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="expected.*shape"):
        _to_btchw_uint8(
            torch.zeros(3, 4, 4),
            image_key="image",
        )


def test_to_btchw_uint8_rejects_invalid_channels():
    with pytest.raises(ValueError, match="1 or 3 channels"):
        _to_btchw_uint8(
            torch.zeros(1, 2, 4, 4, 2),
            image_key="image",
        )


def test_to_btchw_uint8_rejects_empty_trajectory():
    with pytest.raises(
        ValueError,
        match="at least one frame",
    ):
        _to_btchw_uint8(
            torch.zeros(
                1,
                0,
                3,
                4,
                4,
            ),
            image_key="image",
        )


def test_external_trajectory_composites():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
    )

    external = _batched_video([[10, 20, 30]])

    output = step(_transition(external))
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert composites.shape == (
        1,
        3,
        3,
        FRAME_SIZE,
        COMPOSITE_WIDTH,
    )

    # Timestep 0: first=10, previous=10, current=10.
    assert torch.all(composites[:, 0, ..., :FRAME_SIZE] == 10)
    assert torch.all(
        composites[
            :,
            0,
            ...,
            FRAME_SIZE + PADDING : 2 * FRAME_SIZE + PADDING,
        ]
        == 10
    )
    assert torch.all(
        composites[
            :,
            0,
            ...,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 10
    )

    # Timestep 1: first=10, previous=10, current=20.
    assert torch.all(composites[:, 1, ..., :FRAME_SIZE] == 10)
    assert torch.all(
        composites[
            :,
            1,
            ...,
            FRAME_SIZE + PADDING : 2 * FRAME_SIZE + PADDING,
        ]
        == 10
    )
    assert torch.all(
        composites[
            :,
            1,
            ...,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 20
    )

    # Timestep 2: first=10, previous=20, current=30.
    assert torch.all(composites[:, 2, ..., :FRAME_SIZE] == 10)
    assert torch.all(
        composites[
            :,
            2,
            ...,
            FRAME_SIZE + PADDING : 2 * FRAME_SIZE + PADDING,
        ]
        == 20
    )
    assert torch.all(
        composites[
            :,
            2,
            ...,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 30
    )


def test_multiple_trajectories_are_processed_independently():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
    )

    external = _batched_video(
        [
            [10, 20],
            [100, 200],
        ]
    )

    output = step(_transition(external))
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert composites.shape == (
        2,
        2,
        3,
        FRAME_SIZE,
        COMPOSITE_WIDTH,
    )

    assert torch.all(
        composites[
            0,
            1,
            ...,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 20
    )
    assert torch.all(
        composites[
            1,
            1,
            ...,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 200
    )


def test_composite_column_separators_are_black():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
    )

    external = _batched_video([[10, 20]])

    output = step(_transition(external))
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    first_separator = composites[
        ...,
        FRAME_SIZE : FRAME_SIZE + PADDING,
    ]
    second_separator = composites[
        ...,
        2 * FRAME_SIZE + PADDING : 2 * (FRAME_SIZE + PADDING),
    ]

    assert torch.count_nonzero(first_separator) == 0
    assert torch.count_nonzero(second_separator) == 0


def test_dual_view_composite_places_external_above_wrist():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )

    external = _batched_video([[10, 20]])
    wrist = _batched_video([[100, 200]])

    output = step(
        _transition(
            external,
            wrist=wrist,
        )
    )
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert composites.shape == (
        1,
        2,
        3,
        2 * FRAME_SIZE + PADDING,
        COMPOSITE_WIDTH,
    )

    # External row at timestep 1.
    assert torch.all(
        composites[
            :,
            1,
            ...,
            :FRAME_SIZE,
            :FRAME_SIZE,
        ]
        == 10
    )
    assert torch.all(
        composites[
            :,
            1,
            ...,
            :FRAME_SIZE,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 20
    )

    # Black separator between camera rows.
    assert (
        torch.count_nonzero(
            composites[
                ...,
                FRAME_SIZE : FRAME_SIZE + PADDING,
                :,
            ]
        )
        == 0
    )

    # Wrist row at timestep 1.
    assert torch.all(
        composites[
            :,
            1,
            ...,
            FRAME_SIZE + PADDING :,
            :FRAME_SIZE,
        ]
        == 100
    )
    assert torch.all(
        composites[
            :,
            1,
            ...,
            FRAME_SIZE + PADDING :,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 200
    )


def test_processor_is_stateless_between_calls():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
    )

    first_output = step(_transition(_batched_video([[10, 20]])))
    second_output = step(_transition(_batched_video([[100, 200]])))

    first_composites = first_output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]
    second_composites = second_output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert torch.all(first_composites[:, 1, ..., :FRAME_SIZE] == 10)
    assert torch.all(second_composites[:, 1, ..., :FRAME_SIZE] == 100)


def test_missing_observation_raises():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
    )

    transition = {
        TransitionKey.OBSERVATION: None,
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    with pytest.raises(
        ValueError,
        match="observation dictionary",
    ):
        step(transition)


def test_missing_external_image_raises():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
    )

    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    with pytest.raises(
        KeyError,
        match="external image key",
    ):
        step(transition)


def test_missing_configured_wrist_image_raises():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )

    external = _batched_video([[10, 20]])

    with pytest.raises(
        KeyError,
        match="wrist image key",
    ):
        step(_transition(external))


def test_external_and_wrist_dimensions_must_match():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )

    external = _batched_video([[10, 20, 30]])
    wrist = _batched_video([[100, 200]])

    with pytest.raises(
        ValueError,
        match="different external and wrist batch/time dimensions",
    ):
        step(
            _transition(
                external,
                wrist=wrist,
            )
        )


def test_get_config():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
        num_samples=10,
    )

    assert step.get_config() == {
        "external_image_key": EXTERNAL_KEY,
        "wrist_image_key": WRIST_KEY,
        "num_samples": 10,
    }


def test_get_config_preserves_custom_num_samples():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=None,
        num_samples=5,
    )

    assert step.get_config() == {
        "external_image_key": EXTERNAL_KEY,
        "wrist_image_key": None,
        "num_samples": 5,
    }


def test_to_btchw_uint8_accepts_unbatched_channels_last_trajectory():
    images = torch.ones(
        5,
        4,
        6,
        3,
        dtype=torch.uint8,
    )

    result = _to_btchw_uint8(
        images,
        image_key="image",
    )

    assert result.shape == (1, 5, 3, 4, 6)


def test_wrist_only_trajectory_composites():
    step = SOLER1CompositeProcessorStep(
        external_image_key=None,
        wrist_image_key=WRIST_KEY,
    )
    wrist = _batched_video([[10, 20]])
    transition = {
        TransitionKey.OBSERVATION: {
            WRIST_KEY: wrist,
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "pick up the cube",
        },
    }

    output = step(transition)
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert composites.shape == (
        1,
        2,
        3,
        FRAME_SIZE,
        COMPOSITE_WIDTH,
    )
    assert torch.all(composites[:, 1, ..., :FRAME_SIZE] == 10)
    assert torch.all(
        composites[
            :,
            1,
            ...,
            2 * (FRAME_SIZE + PADDING) :,
        ]
        == 20
    )


def test_downsampling_happens_before_composite_construction():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        num_samples=3,
    )

    # Original indices: 0, 1, 2, 3, 4
    # Sampled indices:  0, 2, 4
    external = _batched_video([[10, 20, 30, 40, 50]])

    output = step(_transition(external))
    observation = output[TransitionKey.OBSERVATION]

    composites = observation[SOLER1_COMPOSITE_IMAGE_KEY]
    sample_indices = observation[SOLER1_SAMPLE_INDICES_KEY]
    original_length = observation[SOLER1_ORIGINAL_LENGTH_KEY]

    assert sample_indices.tolist() == [
        0,
        2,
        4,
    ]
    assert original_length.item() == 5
    assert composites.shape == (
        1,
        3,
        3,
        FRAME_SIZE,
        COMPOSITE_WIDTH,
    )

    # Sampled position 1 corresponds to original frame 2.
    # Its previous sampled frame is original frame 0, not frame 1.
    previous_column = composites[
        :,
        1,
        ...,
        FRAME_SIZE + PADDING : 2 * FRAME_SIZE + PADDING,
    ]
    current_column = composites[
        :,
        1,
        ...,
        2 * (FRAME_SIZE + PADDING) :,
    ]

    assert torch.all(previous_column == 10)
    assert torch.all(current_column == 30)

    # Sampled position 2 corresponds to original frame 4.
    # Its previous sampled frame is original frame 2.
    previous_column = composites[
        :,
        2,
        ...,
        FRAME_SIZE + PADDING : 2 * FRAME_SIZE + PADDING,
    ]
    current_column = composites[
        :,
        2,
        ...,
        2 * (FRAME_SIZE + PADDING) :,
    ]

    assert torch.all(previous_column == 30)
    assert torch.all(current_column == 50)


def test_to_btchw_uint8_accepts_batched_single_frames_with_time_dimension():
    images = torch.ones(
        2,
        1,
        3,
        4,
        4,
        dtype=torch.uint8,
    )

    result = _to_btchw_uint8(
        images,
        image_key="image",
    )

    assert result.shape == (2, 1, 3, 4, 4)
    assert result.dtype == torch.uint8


def test_unbatched_tchw_trajectory_is_one_trajectory():
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        num_samples=None,
    )

    trajectory = torch.stack(
        [
            torch.full((3, 4, 4), 10, dtype=torch.uint8),
            torch.full((3, 4, 4), 20, dtype=torch.uint8),
            torch.full((3, 4, 4), 30, dtype=torch.uint8),
        ]
    )

    output = step(_transition(trajectory))
    observation = output[TransitionKey.OBSERVATION]

    composites = observation[SOLER1_COMPOSITE_IMAGE_KEY]
    sample_indices = observation[SOLER1_SAMPLE_INDICES_KEY]
    original_length = observation[SOLER1_ORIGINAL_LENGTH_KEY]

    assert composites.shape == (
        1,
        3,
        3,
        FRAME_SIZE,
        COMPOSITE_WIDTH,
    )
    torch.testing.assert_close(
        sample_indices,
        torch.tensor([0, 1, 2]),
    )
    assert original_length.item() == 3
