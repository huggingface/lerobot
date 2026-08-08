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

"""SOLE-R1 preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    policy_action_to_transition,
)
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.utils.constants import (
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# SOLE-R1 was trained with every individual camera frame resized with
# aspect-ratio-preserving padding to exactly 384 x 384.
FRAME_SIZE = 384
PADDING = 5

# Three temporal frames:
# first | 5-pixel separator | previous | 5-pixel separator | current
COMPOSITE_WIDTH = FRAME_SIZE * 3 + PADDING * 2

# One camera view.
SINGLE_VIEW_COMPOSITE_HEIGHT = FRAME_SIZE

# External view, 5-pixel separator, wrist view.
TWO_VIEW_COMPOSITE_HEIGHT = FRAME_SIZE * 2 + PADDING

SOLER1_FEATURE_PREFIX = f"{OBS_PREFIX}soler1."
SOLER1_COMPOSITE_IMAGE_KEY = f"{SOLER1_FEATURE_PREFIX}composite_image"


def _to_bchw_uint8(images: Tensor | Any, *, image_key: str) -> Tensor:
    """Convert an image or image batch to a CPU ``(B, C, H, W)`` uint8 tensor.

    Accepted input shapes are:

    - ``(C, H, W)``
    - ``(H, W, C)``
    - ``(B, C, H, W)``
    - ``(B, H, W, C)``

    Floating-point inputs in the range ``[0, 1]`` are scaled to ``[0, 255]``.
    """

    tensor = images.detach().cpu() if isinstance(images, Tensor) else torch.as_tensor(images)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(
            f"SOLE-R1 expected {image_key!r} to have shape "
            f"(C,H,W), (H,W,C), (B,C,H,W), or (B,H,W,C); "
            f"got {tuple(tensor.shape)}"
        )

    # Convert channel-last images to channel-first. Restricting valid channel
    # counts to 1 or 3 avoids interpreting an image with height 4 as RGBA.
    if tensor.shape[1] in (1, 3):
        pass
    elif tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"SOLE-R1 expected {image_key!r} to have 1 or 3 channels; got shape {tuple(tensor.shape)}"
        )

    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    if tensor.is_floating_point():
        tensor = tensor.float()
        if tensor.numel() > 0 and tensor.max().item() <= 1.0:
            tensor = tensor * 255.0

    return tensor.clamp(0, 255).round().to(torch.uint8).contiguous()


def resize_with_padding(image: Tensor) -> Tensor:
    """Resize one ``(3, H, W)`` frame to ``(3, 384, 384)``.

    The aspect ratio is preserved. The resized frame is centered on a black
    384 x 384 canvas, matching the SOLE-R1 preprocessing used in RewardGen.
    """

    if image.ndim != 3:
        raise ValueError(f"SOLE-R1 expected one frame with shape (C,H,W); got {tuple(image.shape)}")

    channels, height, width = image.shape

    if channels != 3:
        raise ValueError(f"SOLE-R1 expected an RGB frame with 3 channels; got {channels}")

    if height <= 0 or width <= 0:
        raise ValueError(f"SOLE-R1 expected positive image dimensions; got {(height, width)}")

    scale = FRAME_SIZE / max(height, width)
    resized_height = max(1, int(height * scale))
    resized_width = max(1, int(width * scale))

    resized = functional.interpolate(
        image.unsqueeze(0).float(),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    resized = resized.round().clamp(0, 255).to(torch.uint8)

    output = torch.zeros(
        (3, FRAME_SIZE, FRAME_SIZE),
        dtype=torch.uint8,
    )

    y_offset = (FRAME_SIZE - resized_height) // 2
    x_offset = (FRAME_SIZE - resized_width) // 2

    output[
        :,
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    return output


def compose_temporal_frames(
    first_frame: Tensor,
    previous_frame: Tensor,
    current_frame: Tensor,
) -> Tensor:
    """Create a single-camera SOLE-R1 temporal row.

    The output layout is:

    ``first | previous | current``

    Each frame is 384 x 384 and adjacent frames are separated by five black
    pixels. The returned tensor has shape ``(3, 384, 1162)``.
    """

    first = resize_with_padding(first_frame)
    previous = resize_with_padding(previous_frame)
    current = resize_with_padding(current_frame)

    separator = torch.zeros(
        (3, FRAME_SIZE, PADDING),
        dtype=torch.uint8,
    )

    composite = torch.cat(
        [
            first,
            separator,
            previous,
            separator,
            current,
        ],
        dim=2,
    )

    expected_shape = (
        3,
        SINGLE_VIEW_COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
    )

    if tuple(composite.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected SOLE-R1 temporal composite shape {tuple(composite.shape)}; expected {expected_shape}"
        )

    return composite


def compose_camera_views(
    external_row: Tensor,
    wrist_row: Tensor | None = None,
) -> Tensor:
    """Combine the external and optional wrist temporal rows.

    For one camera, the returned shape is ``(3, 384, 1162)``.

    For external and wrist cameras, the external row is placed on top and the
    wrist row on the bottom, separated by five black pixels. The returned shape
    is ``(3, 773, 1162)``.
    """

    expected_row_shape = (
        3,
        SINGLE_VIEW_COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
    )

    if tuple(external_row.shape) != expected_row_shape:
        raise ValueError(
            f"Unexpected external row shape {tuple(external_row.shape)}; expected {expected_row_shape}"
        )

    if wrist_row is None:
        return external_row

    if tuple(wrist_row.shape) != expected_row_shape:
        raise ValueError(
            f"Unexpected wrist row shape {tuple(wrist_row.shape)}; expected {expected_row_shape}"
        )

    separator = torch.zeros(
        (3, PADDING, COMPOSITE_WIDTH),
        dtype=torch.uint8,
    )

    composite = torch.cat(
        [
            external_row,
            separator,
            wrist_row,
        ],
        dim=1,
    )

    expected_shape = (
        3,
        TWO_VIEW_COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
    )

    if tuple(composite.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected SOLE-R1 two-view composite shape {tuple(composite.shape)}; expected {expected_shape}"
        )

    return composite


def create_composite_frame(
    *,
    first_external_frame: Tensor,
    previous_external_frame: Tensor,
    current_external_frame: Tensor,
    first_wrist_frame: Tensor | None = None,
    previous_wrist_frame: Tensor | None = None,
    current_wrist_frame: Tensor | None = None,
) -> Tensor:
    """Create one complete SOLE-R1 composite frame."""

    wrist_frames = (
        first_wrist_frame,
        previous_wrist_frame,
        current_wrist_frame,
    )

    if any(frame is not None for frame in wrist_frames) and not all(
        frame is not None for frame in wrist_frames
    ):
        raise ValueError("SOLE-R1 requires either all three wrist frames or no wrist frames")

    external_row = compose_temporal_frames(
        first_frame=first_external_frame,
        previous_frame=previous_external_frame,
        current_frame=current_external_frame,
    )

    wrist_row = None
    if all(frame is not None for frame in wrist_frames):
        assert first_wrist_frame is not None
        assert previous_wrist_frame is not None
        assert current_wrist_frame is not None

        wrist_row = compose_temporal_frames(
            first_frame=first_wrist_frame,
            previous_frame=previous_wrist_frame,
            current_frame=current_wrist_frame,
        )

    return compose_camera_views(
        external_row=external_row,
        wrist_row=wrist_row,
    )


@dataclass
@ProcessorStepRegistry.register(name="soler1_composite")
class SOLER1CompositeProcessorStep(ProcessorStep):
    """Build SOLE-R1's first/previous/current composite image.

    This processor is stateful. On the first call after ``reset()``, the
    current frame is used for the first, previous, and current columns. On
    later calls, the original first frame and the most recent previous frame
    are retained.

    Call ``preprocessor.reset()`` at the beginning of each episode.
    """

    external_image_key: str
    wrist_image_key: str | None = None

    _first_external_frames: Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _previous_external_frames: Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _first_wrist_frames: Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _previous_wrist_frames: Tensor | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def reset(self) -> None:
        """Clear all episode-local frame history."""

        self._first_external_frames = None
        self._previous_external_frames = None
        self._first_wrist_frames = None
        self._previous_wrist_frames = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("SOLE-R1 preprocessing requires an observation")

        if self.external_image_key not in observation:
            raise KeyError(
                f"SOLE-R1 expected external image key {self.external_image_key!r} in the observation"
            )

        current_external_frames = _to_bchw_uint8(
            observation[self.external_image_key],
            image_key=self.external_image_key,
        )

        current_wrist_frames: Tensor | None = None
        if self.wrist_image_key is not None:
            if self.wrist_image_key not in observation:
                raise KeyError(
                    f"SOLE-R1 expected wrist image key {self.wrist_image_key!r} in the observation"
                )

            current_wrist_frames = _to_bchw_uint8(
                observation[self.wrist_image_key],
                image_key=self.wrist_image_key,
            )

            if current_wrist_frames.shape[0] != current_external_frames.shape[0]:
                raise ValueError(
                    "SOLE-R1 received different external and wrist batch "
                    f"sizes: {current_external_frames.shape[0]} and "
                    f"{current_wrist_frames.shape[0]}"
                )

        batch_size = current_external_frames.shape[0]

        if self._first_external_frames is None:
            self._first_external_frames = current_external_frames.clone()
            self._previous_external_frames = current_external_frames.clone()

            if current_wrist_frames is not None:
                self._first_wrist_frames = current_wrist_frames.clone()
                self._previous_wrist_frames = current_wrist_frames.clone()
        else:
            if self._first_external_frames.shape[0] != batch_size:
                raise ValueError(
                    "SOLE-R1 processor batch size changed within an episode. "
                    "Call preprocessor.reset() before starting a new episode."
                )

            if self.wrist_image_key is not None and self._first_wrist_frames is None:
                raise RuntimeError(
                    "SOLE-R1 wrist-camera state is inconsistent. Call preprocessor.reset() before continuing."
                )

        assert self._first_external_frames is not None
        assert self._previous_external_frames is not None

        composite_frames: list[Tensor] = []

        for batch_index in range(batch_size):
            first_wrist_frame = None
            previous_wrist_frame = None
            current_wrist_frame = None

            if current_wrist_frames is not None:
                assert self._first_wrist_frames is not None
                assert self._previous_wrist_frames is not None

                first_wrist_frame = self._first_wrist_frames[batch_index]
                previous_wrist_frame = self._previous_wrist_frames[batch_index]
                current_wrist_frame = current_wrist_frames[batch_index]

            composite_frames.append(
                create_composite_frame(
                    first_external_frame=self._first_external_frames[batch_index],
                    previous_external_frame=self._previous_external_frames[batch_index],
                    current_external_frame=current_external_frames[batch_index],
                    first_wrist_frame=first_wrist_frame,
                    previous_wrist_frame=previous_wrist_frame,
                    current_wrist_frame=current_wrist_frame,
                )
            )

        composite_batch = torch.stack(composite_frames, dim=0)

        # Save the current frames as the previous frames for the next call.
        self._previous_external_frames = current_external_frames.clone()

        if current_wrist_frames is not None:
            self._previous_wrist_frames = current_wrist_frames.clone()

        new_observation = dict(observation)
        new_observation[SOLER1_COMPOSITE_IMAGE_KEY] = composite_batch

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self,
        features: dict[
            PipelineFeatureType,
            dict[str, PolicyFeature],
        ],
    ) -> dict[
        PipelineFeatureType,
        dict[str, PolicyFeature],
    ]:
        composite_height = (
            TWO_VIEW_COMPOSITE_HEIGHT if self.wrist_image_key is not None else SINGLE_VIEW_COMPOSITE_HEIGHT
        )

        features[PipelineFeatureType.OBSERVATION][SOLER1_COMPOSITE_IMAGE_KEY] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, composite_height, COMPOSITE_WIDTH),
        )

        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "external_image_key": self.external_image_key,
            "wrist_image_key": self.wrist_image_key,
        }


def make_soler1_pre_post_processors(
    config: SOLER1Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create the SOLE-R1 preprocessing and postprocessing pipelines."""

    del dataset_stats

    preprocessor = PolicyProcessorPipeline[
        dict[str, Any],
        dict[str, Any],
    ](
        steps=[
            AddBatchDimensionProcessorStep(),
            SOLER1CompositeProcessorStep(
                external_image_key=config.external_image_key,
                wrist_image_key=config.wrist_image_key,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    postprocessor = PolicyProcessorPipeline[
        PolicyAction,
        PolicyAction,
    ](
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )

    return preprocessor, postprocessor
