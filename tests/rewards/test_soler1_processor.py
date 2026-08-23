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

"""Tests for SOLE-R1 static preprocessing; no reward-model weights are loaded."""

from __future__ import annotations

from typing import Any

import cv2
import pytest
import torch
from torch import Tensor

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.processor_soler1 import (
    COMPOSITE_WIDTH,
    FRAME_SIZE,
    PADDING,
    SINGLE_VIEW_COMPOSITE_HEIGHT,
    SOLER1_COMPOSITE_IMAGE_KEY,
    SOLER1_IMAGE_GRID_THW_KEY,
    SOLER1_IMAGE_TOKEN_COUNT_KEY,
    SOLER1_PIXEL_VALUES_KEY,
    TWO_VIEW_COMPOSITE_HEIGHT,
    SOLER1CompositeProcessorStep,
    SOLER1ImageEncoderProcessorStep,
    _composite_to_smart_resized_pil,
    _to_btchw_uint8,
    make_soler1_pre_post_processors,
    resize_with_padding,
)
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME
from tests.utils import skip_if_package_missing

EXTERNAL_KEY = "observation.images.external"
WRIST_KEY = "observation.images.wrist"


def _video(
    batch_size: int = 1,
    timesteps: int = 3,
    *,
    height: int = 32,
    width: int = 48,
    offset: int = 0,
) -> Tensor:
    values = torch.arange(
        offset,
        offset + batch_size * timesteps,
        dtype=torch.uint8,
    ).reshape(batch_size, timesteps)
    return values[:, :, None, None, None].repeat(1, 1, 3, height, width)


def _transition(
    *,
    external: Tensor | None = None,
    wrist: Tensor | None = None,
) -> EnvTransition:
    observation: dict[str, Any] = {}
    if external is not None:
        observation[EXTERNAL_KEY] = external
    if wrist is not None:
        observation[WRIST_KEY] = wrist
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.COMPLEMENTARY_DATA: {"task": "pick up the cube"},
    }


def test_canonical_btchw_input_preserves_batches_and_all_timesteps() -> None:
    step = SOLER1CompositeProcessorStep(external_image_key=EXTERNAL_KEY)

    output = step(_transition(external=_video(batch_size=2, timesteps=3)))
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert composites.shape == (2, 3, 3, SINGLE_VIEW_COMPOSITE_HEIGHT, COMPOSITE_WIDTH)
    # Inspect the center pixel so letterbox padding does not affect the assertion.
    center = FRAME_SIZE // 2
    previous_x = FRAME_SIZE + PADDING + center
    current_x = 2 * (FRAME_SIZE + PADDING) + center
    assert composites[0, 2, 0, center, previous_x].item() == 1
    assert composites[0, 2, 0, center, current_x].item() == 2
    assert composites[1, 2, 0, center, previous_x].item() == 4
    assert composites[1, 2, 0, center, current_x].item() == 5


def test_rank_four_is_rejected_as_ambiguous() -> None:
    images = torch.zeros(3, 3, 32, 48, dtype=torch.uint8)

    with pytest.raises(ValueError, match="ambiguous rank-4"):
        _to_btchw_uint8(images, image_key=EXTERNAL_KEY)


def test_channels_last_is_rejected_by_canonical_contract() -> None:
    images = torch.zeros(1, 2, 32, 48, 3, dtype=torch.uint8)

    with pytest.raises(ValueError, match="channels-first"):
        _to_btchw_uint8(images, image_key=EXTERNAL_KEY)


def test_external_and_wrist_batch_time_dimensions_must_match() -> None:
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )

    with pytest.raises(ValueError, match="different external and wrist batch/time dimensions"):
        step(
            _transition(
                external=_video(batch_size=1, timesteps=3),
                wrist=_video(batch_size=1, timesteps=2),
            )
        )


@pytest.mark.parametrize(
    ("external_key", "wrist_key", "external", "wrist", "expected_height"),
    [
        (EXTERNAL_KEY, None, _video(timesteps=2, offset=10), None, SINGLE_VIEW_COMPOSITE_HEIGHT),
        (None, WRIST_KEY, None, _video(timesteps=2, offset=20), SINGLE_VIEW_COMPOSITE_HEIGHT),
        (
            EXTERNAL_KEY,
            WRIST_KEY,
            _video(timesteps=2, offset=10),
            _video(timesteps=2, offset=20),
            TWO_VIEW_COMPOSITE_HEIGHT,
        ),
    ],
)
def test_supported_camera_modes_have_expected_composite_dimensions(
    external_key: str | None,
    wrist_key: str | None,
    external: Tensor | None,
    wrist: Tensor | None,
    expected_height: int,
) -> None:
    step = SOLER1CompositeProcessorStep(
        external_image_key=external_key,
        wrist_image_key=wrist_key,
    )

    output = step(_transition(external=external, wrist=wrist))
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]

    assert composites.shape == (1, 2, 3, expected_height, COMPOSITE_WIDTH)


def test_external_and_wrist_composite_order() -> None:
    external = torch.full((1, 2, 3, 32, 32), 10, dtype=torch.uint8)
    wrist = torch.full((1, 2, 3, 32, 32), 20, dtype=torch.uint8)
    step = SOLER1CompositeProcessorStep(
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )

    output = step(_transition(external=external, wrist=wrist))
    composites = output[TransitionKey.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY]
    panel_centers = (
        FRAME_SIZE // 2,
        FRAME_SIZE + PADDING + FRAME_SIZE // 2,
        2 * (FRAME_SIZE + PADDING) + FRAME_SIZE // 2,
    )
    external_center_y = FRAME_SIZE // 2
    wrist_center_y = FRAME_SIZE + PADDING + FRAME_SIZE // 2

    for center_x in panel_centers:
        assert torch.all(composites[:, :, :, external_center_y, center_x] == 10)
        assert torch.all(composites[:, :, :, wrist_center_y, center_x] == 20)

    assert torch.all(composites[:, :, :, FRAME_SIZE : FRAME_SIZE + PADDING, :] == 0)


def test_no_configured_camera_is_rejected() -> None:
    step = SOLER1CompositeProcessorStep(external_image_key=None, wrist_image_key=None)

    with pytest.raises(ValueError, match="at least one camera"):
        step(_transition())


@pytest.mark.parametrize(
    ("external_key", "wrist_key", "transition", "expected_message"),
    [
        (EXTERNAL_KEY, None, _transition(wrist=_video()), "external image key"),
        (None, WRIST_KEY, _transition(external=_video()), "wrist image key"),
    ],
)
def test_missing_configured_camera_is_rejected(
    external_key: str | None,
    wrist_key: str | None,
    transition: EnvTransition,
    expected_message: str,
) -> None:
    step = SOLER1CompositeProcessorStep(
        external_image_key=external_key,
        wrist_image_key=wrist_key,
    )

    with pytest.raises(KeyError, match=expected_message):
        step(transition)


def test_letterbox_uses_opencv_inter_area(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1 import processor_soler1

    interpolation_values: list[int] = []
    original_resize = processor_soler1.cv2.resize

    def _record_resize(*args: Any, **kwargs: Any) -> Any:
        interpolation_values.append(kwargs["interpolation"])
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(processor_soler1.cv2, "resize", _record_resize)

    resized = resize_with_padding(torch.zeros(3, 32, 48, dtype=torch.uint8))

    assert resized.shape == (3, FRAME_SIZE, FRAME_SIZE)
    assert interpolation_values == [cv2.INTER_AREA]


@skip_if_package_missing("qwen-vl-utils", import_name="qwen_vl_utils")
def test_post_composite_smart_resize_matches_public_dimensions() -> None:
    single_view = torch.zeros(
        3,
        SINGLE_VIEW_COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
        dtype=torch.uint8,
    )
    two_views = torch.zeros(
        3,
        TWO_VIEW_COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
        dtype=torch.uint8,
    )

    assert _composite_to_smart_resized_pil(
        single_view,
        factor=28,
        min_pixels=3136,
        max_pixels=12845056,
    ).size == (1176, 392)
    assert _composite_to_smart_resized_pil(
        two_views,
        factor=28,
        min_pixels=3136,
        max_pixels=12845056,
    ).size == (1176, 784)


class _FakeImageProcessor:
    merge_size = 2

    def __call__(self, *, images: list[Any], return_tensors: str) -> dict[str, Tensor]:
        assert return_tensors == "pt"
        grids = [[1, 24 if image.height == 392 else 48, 74] for image in images]
        image_grid_thw = torch.tensor(grids, dtype=torch.long)
        patch_count = int(image_grid_thw.prod(dim=-1).sum().item())
        return {
            "pixel_values": torch.zeros(patch_count, 8),
            "image_grid_thw": image_grid_thw,
        }


class _FakeAutoProcessor:
    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeAutoProcessor:  # noqa: ARG003
        return cls()


def _patch_auto_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.rewards.soler1 import processor_soler1

    monkeypatch.setattr(processor_soler1, "AutoProcessor", _FakeAutoProcessor)


@skip_if_package_missing("transformers")
@skip_if_package_missing("qwen-vl-utils", import_name="qwen_vl_utils")
@pytest.mark.parametrize(
    ("external_key", "wrist_key", "expected_grid"),
    [
        (EXTERNAL_KEY, None, [1, 24, 74]),
        (None, WRIST_KEY, [1, 24, 74]),
        (EXTERNAL_KEY, WRIST_KEY, [1, 48, 74]),
    ],
)
def test_static_preprocessing_without_loading_reward_model(
    monkeypatch: pytest.MonkeyPatch,
    external_key: str | None,
    wrist_key: str | None,
    expected_grid: list[int],
) -> None:
    _patch_auto_processor(monkeypatch)
    external = _video(batch_size=2, timesteps=3) if external_key is not None else None
    wrist = _video(batch_size=2, timesteps=3, offset=20) if wrist_key is not None else None
    composite_step = SOLER1CompositeProcessorStep(
        external_image_key=external_key,
        wrist_image_key=wrist_key,
    )
    encoder_step = SOLER1ImageEncoderProcessorStep(model_name="fake")

    output = encoder_step(composite_step(_transition(external=external, wrist=wrist)))
    observation = output[TransitionKey.OBSERVATION]

    assert SOLER1_COMPOSITE_IMAGE_KEY not in observation
    assert observation[SOLER1_IMAGE_GRID_THW_KEY].shape == (2, 3, 3)
    assert observation[SOLER1_IMAGE_GRID_THW_KEY][0, 0].tolist() == expected_grid
    assert observation[SOLER1_PIXEL_VALUES_KEY].shape[:2] == (2, 3)
    assert observation[SOLER1_IMAGE_TOKEN_COUNT_KEY].shape == (2, 3)
    assert (
        observation[SOLER1_IMAGE_TOKEN_COUNT_KEY][0, 0].item()
        == torch.tensor(expected_grid).prod().item() // 4
    )


@skip_if_package_missing("transformers")
@skip_if_package_missing("qwen-vl-utils", import_name="qwen_vl_utils")
def test_processor_serialization_and_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_auto_processor(monkeypatch)
    config = SOLER1Config(
        device="cpu",
        external_image_key=EXTERNAL_KEY,
        wrist_image_key=WRIST_KEY,
    )
    preprocessor, _ = make_soler1_pre_post_processors(config)

    preprocessor.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )

    assert [type(step) for step in loaded.steps] == [type(step) for step in preprocessor.steps]
    assert loaded.steps[0].get_config() == preprocessor.steps[0].get_config()
    assert loaded.steps[1].get_config() == preprocessor.steps[1].get_config()
