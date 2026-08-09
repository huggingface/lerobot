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

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
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

# SOLE-R1 was trained with each camera frame resized with
# aspect-ratio-preserving padding to exactly 384 x 384.
FRAME_SIZE = 384
PADDING = 5

# first | separator | previous | separator | current
COMPOSITE_WIDTH = FRAME_SIZE * 3 + PADDING * 2

SINGLE_VIEW_COMPOSITE_HEIGHT = FRAME_SIZE

# External view, separator, wrist view.
TWO_VIEW_COMPOSITE_HEIGHT = FRAME_SIZE * 2 + PADDING

SOLER1_FEATURE_PREFIX = f"{OBS_PREFIX}soler1."
SOLER1_COMPOSITE_IMAGE_KEY = f"{SOLER1_FEATURE_PREFIX}composite_image"
SOLER1_SAMPLE_INDICES_KEY = f"{SOLER1_FEATURE_PREFIX}sample_indices"
SOLER1_ORIGINAL_LENGTH_KEY = f"{SOLER1_FEATURE_PREFIX}original_length"


def _sample_frame_indices(
    trajectory_length: int,
    downsample_to: int | None,
) -> Tensor:
    """Return uniformly spaced frame indices.

    When downsampling selects at least two frames, the first and final
    trajectory frames are included.
    """

    if trajectory_length < 1:
        raise ValueError("SOLE-R1 requires at least one frame per trajectory")

    if downsample_to is not None and downsample_to < 1:
        raise ValueError(f"downsample_to must be >= 1 or None, got {downsample_to}")

    if downsample_to is None or trajectory_length <= downsample_to:
        return torch.arange(
            trajectory_length,
            dtype=torch.long,
        )

    return torch.linspace(
        0,
        trajectory_length - 1,
        steps=downsample_to,
    ).long()


def _to_btchw_uint8(
    images: Tensor | Any,
    *,
    image_key: str,
) -> Tensor:
    """Convert camera frames to a CPU ``(B,T,C,H,W)`` uint8 tensor.

    Canonical input shapes are:

    - ``(T,H,W,C)`` for an unbatched trajectory;
    - ``(B,T,H,W,C)`` for batched trajectories.

    Batched channels-first LeRobot tensors remain supported for
    compatibility. Floating-point inputs in ``[0,1]`` are scaled to
    ``[0,255]``.
    """

    tensor = images.detach().cpu() if isinstance(images, Tensor) else torch.as_tensor(images)

    if tensor.ndim == 4:
        if tensor.shape[-1] in (1, 3):
            # Unbatched channels-last trajectory: (T,H,W,C).
            tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        elif tensor.shape[1] in (1, 3):
            # Backward-compatible batched single frames: (B,C,H,W).
            tensor = tensor.unsqueeze(1)
        else:
            raise ValueError(
                f"SOLE-R1 expected {image_key!r} to have 1 or 3 channels; got shape {tuple(tensor.shape)}"
            )
    elif tensor.ndim == 5:
        if tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 1, 4, 2, 3)
        elif tensor.shape[2] not in (1, 3):
            raise ValueError(
                f"SOLE-R1 expected {image_key!r} to have 1 or 3 channels; got shape {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(
            f"SOLE-R1 expected {image_key!r} to have shape "
            "(T,H,W,C) or (B,T,H,W,C); "
            f"got {tuple(tensor.shape)}"
        )

    if tensor.shape[2] == 1:
        tensor = tensor.repeat(1, 1, 3, 1, 1)

    if tensor.shape[1] < 1:
        raise ValueError("SOLE-R1 requires at least one frame per trajectory")

    if tensor.is_floating_point():
        tensor = tensor.float()
        if tensor.numel() > 0 and tensor.max().item() <= 1.0:
            tensor = tensor * 255.0

    return tensor.clamp(0, 255).round().to(torch.uint8).contiguous()


def resize_with_padding(image: Tensor) -> Tensor:
    """Resize one ``(3,H,W)`` frame to ``(3,384,384)``.

    The aspect ratio is preserved. The resized image is centered on a
    black canvas, matching the preprocessing used by RewardGen.
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

    Each frame is 384 x 384. Adjacent frames are separated by five
    black pixels. The output shape is ``(3,384,1162)``.
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
    external_row: Tensor | None = None,
    wrist_row: Tensor | None = None,
) -> Tensor:
    """Combine optional external and wrist temporal rows.

    A single-camera composite has shape ``(3,384,1162)``.

    With both cameras, the external row is placed above the wrist row,
    separated by five black pixels. The result has shape
    ``(3,773,1162)``.
    """

    expected_row_shape = (
        3,
        SINGLE_VIEW_COMPOSITE_HEIGHT,
        COMPOSITE_WIDTH,
    )

    if external_row is None and wrist_row is None:
        raise ValueError("SOLE-R1 requires at least one camera view")

    if external_row is not None and tuple(external_row.shape) != expected_row_shape:
        raise ValueError(
            f"Unexpected external row shape {tuple(external_row.shape)}; expected {expected_row_shape}"
        )

    if wrist_row is None:
        assert external_row is not None
        return external_row

    if tuple(wrist_row.shape) != expected_row_shape:
        raise ValueError(
            f"Unexpected wrist row shape {tuple(wrist_row.shape)}; expected {expected_row_shape}"
        )

    if external_row is None:
        return wrist_row

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
    first_external_frame: Tensor | None = None,
    previous_external_frame: Tensor | None = None,
    current_external_frame: Tensor | None = None,
    first_wrist_frame: Tensor | None = None,
    previous_wrist_frame: Tensor | None = None,
    current_wrist_frame: Tensor | None = None,
) -> Tensor:
    """Create one complete SOLE-R1 composite image."""

    external_frames = (
        first_external_frame,
        previous_external_frame,
        current_external_frame,
    )
    wrist_frames = (
        first_wrist_frame,
        previous_wrist_frame,
        current_wrist_frame,
    )

    for view_name, frames in (
        ("external", external_frames),
        ("wrist", wrist_frames),
    ):
        if any(frame is not None for frame in frames) and not all(frame is not None for frame in frames):
            raise ValueError(f"SOLE-R1 requires either all three {view_name} frames or no {view_name} frames")

    if not any(frame is not None for frame in (*external_frames, *wrist_frames)):
        raise ValueError("SOLE-R1 requires at least one camera view")

    external_row = None
    if all(frame is not None for frame in external_frames):
        assert first_external_frame is not None
        assert previous_external_frame is not None
        assert current_external_frame is not None

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
    """Build SOLE-R1 composites after downsampling raw trajectories.

    Camera inputs use ``(B,T,H,W,C)`` or, without a batch dimension,
    ``(T,H,W,C)``.

    Downsampling is applied to the raw camera trajectories before any
    composite images are constructed. Therefore, each composite contains:

    - the first frame of the original trajectory;
    - the previous sampled frame;
    - the current sampled frame.

    The output composite tensor uses shape ``(B,S,C,H,W)``, where ``S``
    is the number of sampled frames.
    """

    external_image_key: str | None = None
    wrist_image_key: str | None = None
    downsample_to: int | None = 10

    def reset(self) -> None:
        """No-op because complete trajectories are processed per call."""

    def __call__(
        self,
        transition: EnvTransition,
    ) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)

        if not isinstance(observation, dict):
            raise ValueError("SOLE-R1 preprocessing requires an observation dictionary")

        if self.external_image_key is None and self.wrist_image_key is None:
            raise ValueError("SOLE-R1 requires at least one camera view")

        external_videos: Tensor | None = None

        if self.external_image_key is not None:
            if self.external_image_key not in observation:
                raise KeyError(
                    f"SOLE-R1 expected external image key {self.external_image_key!r} in the observation"
                )

            external_videos = _to_btchw_uint8(
                observation[self.external_image_key],
                image_key=self.external_image_key,
            )

        wrist_videos: Tensor | None = None

        if self.wrist_image_key is not None:
            if self.wrist_image_key not in observation:
                raise KeyError(
                    f"SOLE-R1 expected wrist image key {self.wrist_image_key!r} in the observation"
                )

            wrist_videos = _to_btchw_uint8(
                observation[self.wrist_image_key],
                image_key=self.wrist_image_key,
            )

        if (
            external_videos is not None
            and wrist_videos is not None
            and external_videos.shape[:2] != wrist_videos.shape[:2]
        ):
            raise ValueError(
                "SOLE-R1 received different external and wrist "
                "batch/time dimensions: "
                f"{tuple(external_videos.shape[:2])} and "
                f"{tuple(wrist_videos.shape[:2])}"
            )

        reference_videos = external_videos if external_videos is not None else wrist_videos
        assert reference_videos is not None

        batch_size, original_length = reference_videos.shape[:2]

        sample_indices = _sample_frame_indices(
            original_length,
            self.downsample_to,
        )
        sampled_indices = sample_indices.tolist()

        batch_composites: list[Tensor] = []

        for batch_index in range(batch_size):
            trajectory_composites: list[Tensor] = []

            for sampled_position, current_timestep in enumerate(sampled_indices):
                previous_position = max(
                    sampled_position - 1,
                    0,
                )
                previous_timestep = sampled_indices[previous_position]

                first_external_frame = None
                previous_external_frame = None
                current_external_frame = None

                if external_videos is not None:
                    first_external_frame = external_videos[
                        batch_index,
                        0,
                    ]
                    previous_external_frame = external_videos[
                        batch_index,
                        previous_timestep,
                    ]
                    current_external_frame = external_videos[
                        batch_index,
                        current_timestep,
                    ]

                first_wrist_frame = None
                previous_wrist_frame = None
                current_wrist_frame = None

                if wrist_videos is not None:
                    first_wrist_frame = wrist_videos[
                        batch_index,
                        0,
                    ]
                    previous_wrist_frame = wrist_videos[
                        batch_index,
                        previous_timestep,
                    ]
                    current_wrist_frame = wrist_videos[
                        batch_index,
                        current_timestep,
                    ]

                composite = create_composite_frame(
                    first_external_frame=first_external_frame,
                    previous_external_frame=previous_external_frame,
                    current_external_frame=current_external_frame,
                    first_wrist_frame=first_wrist_frame,
                    previous_wrist_frame=previous_wrist_frame,
                    current_wrist_frame=current_wrist_frame,
                )

                trajectory_composites.append(composite)

            batch_composites.append(
                torch.stack(
                    trajectory_composites,
                    dim=0,
                )
            )

        composite_batch = torch.stack(
            batch_composites,
            dim=0,
        )

        new_observation = dict(observation)
        new_observation[SOLER1_COMPOSITE_IMAGE_KEY] = composite_batch
        new_observation[SOLER1_SAMPLE_INDICES_KEY] = sample_indices
        new_observation[SOLER1_ORIGINAL_LENGTH_KEY] = torch.tensor(
            original_length,
            dtype=torch.long,
        )

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
        # The composite has a dynamic sampled-time dimension.
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "external_image_key": self.external_image_key,
            "wrist_image_key": self.wrist_image_key,
            "downsample_to": self.downsample_to,
        }


def make_soler1_pre_post_processors(
    config: SOLER1Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[
        dict[str, Any],
        dict[str, Any],
    ],
    PolicyProcessorPipeline[
        PolicyAction,
        PolicyAction,
    ],
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
                downsample_to=config.downsample_to,
            ),
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
