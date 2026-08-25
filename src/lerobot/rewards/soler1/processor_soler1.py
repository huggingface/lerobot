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

"""Serializable static preprocessing for SOLE-R1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
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
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

FRAME_SIZE = 384
PADDING = 5
COMPOSITE_WIDTH = FRAME_SIZE * 3 + PADDING * 2
SINGLE_VIEW_COMPOSITE_HEIGHT = FRAME_SIZE
TWO_VIEW_COMPOSITE_HEIGHT = FRAME_SIZE * 2 + PADDING

SOLER1_FEATURE_PREFIX = f"{OBS_PREFIX}soler1."
SOLER1_COMPOSITE_IMAGE_KEY = f"{SOLER1_FEATURE_PREFIX}composite_image"
SOLER1_PIXEL_VALUES_KEY = f"{SOLER1_FEATURE_PREFIX}pixel_values"
SOLER1_IMAGE_GRID_THW_KEY = f"{SOLER1_FEATURE_PREFIX}image_grid_thw"
SOLER1_IMAGE_TOKEN_COUNT_KEY = f"{SOLER1_FEATURE_PREFIX}image_token_count"


def _to_btchw_uint8(images: Tensor | Any, *, image_key: str) -> Tensor:
    """Validate and convert canonical ``(B,T,C,H,W)`` input to CPU uint8.

    Rank four is deliberately rejected because it is ambiguous between
    ``(B,C,H,W)`` and ``(T,C,H,W)``. Callers must add the missing batch or
    time axis explicitly before invoking the SOLE-R1 pipeline.
    """

    tensor = images.detach().cpu() if isinstance(images, Tensor) else torch.as_tensor(images)
    if tensor.ndim == 4:
        raise ValueError(
            f"SOLE-R1 received ambiguous rank-4 input for {image_key!r} with shape "
            f"{tuple(tensor.shape)}; use canonical (B,T,C,H,W), including an explicit "
            "batch and time dimension"
        )
    if tensor.ndim != 5:
        raise ValueError(
            f"SOLE-R1 expected {image_key!r} with canonical shape (B,T,C,H,W); got {tuple(tensor.shape)}"
        )
    if tensor.shape[2] not in (1, 3):
        if tensor.shape[-1] in (1, 3):
            raise ValueError(
                f"SOLE-R1 expected channels-first (B,T,C,H,W) input for {image_key!r}; "
                f"got channels-last shape {tuple(tensor.shape)}"
            )
        raise ValueError(f"SOLE-R1 expected 1 or 3 channels for {image_key!r}; got {tuple(tensor.shape)}")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError("SOLE-R1 requires at least one batch element and one timestep")
    if tensor.shape[-2] < 1 or tensor.shape[-1] < 1:
        raise ValueError("SOLE-R1 requires positive image height and width")

    if tensor.shape[2] == 1:
        tensor = tensor.repeat(1, 1, 3, 1, 1)
    if tensor.is_floating_point():
        tensor = tensor.float()
        if tensor.numel() and tensor.max().item() <= 1.0:
            tensor = tensor * 255.0
    return tensor.clamp(0, 255).round().to(torch.uint8).contiguous()


def resize_with_padding(image: Tensor) -> Tensor:
    """Letterbox one RGB frame to 384 x 384 with OpenCV ``INTER_AREA``.

    This intentionally matches the public SOLE-R1 server, including integer
    truncation of the resized dimensions.
    """

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"SOLE-R1 expected one RGB frame with shape (3,H,W); got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if height < 1 or width < 1:
        raise ValueError(f"SOLE-R1 expected positive image dimensions; got {(height, width)}")

    array = image.permute(1, 2, 0).contiguous().cpu().numpy()
    scale = FRAME_SIZE / max(height, width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    output = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    y_offset = (FRAME_SIZE - resized_height) // 2
    x_offset = (FRAME_SIZE - resized_width) // 2
    output[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return torch.from_numpy(output).permute(2, 0, 1).contiguous()


def compose_temporal_frames(first_frame: Tensor, previous_frame: Tensor, current_frame: Tensor) -> Tensor:
    """Return ``first | previous | current`` with shape ``(3,384,1162)``."""

    separator = torch.zeros((3, FRAME_SIZE, PADDING), dtype=torch.uint8)
    return torch.cat(
        [
            resize_with_padding(first_frame),
            separator,
            resize_with_padding(previous_frame),
            separator,
            resize_with_padding(current_frame),
        ],
        dim=2,
    )


def compose_camera_views(
    external_row: Tensor | None = None,
    wrist_row: Tensor | None = None,
) -> Tensor:
    """Return an external-only, wrist-only, or external+wrist composite."""

    expected = (3, SINGLE_VIEW_COMPOSITE_HEIGHT, COMPOSITE_WIDTH)

    if external_row is None and wrist_row is None:
        raise ValueError("SOLE-R1 requires at least one camera view")

    if external_row is not None and tuple(external_row.shape) != expected:
        raise ValueError(f"Unexpected external row shape {tuple(external_row.shape)}; expected {expected}")

    if external_row is None:
        assert wrist_row is not None
        if tuple(wrist_row.shape) != expected:
            raise ValueError(f"Unexpected wrist row shape {tuple(wrist_row.shape)}; expected {expected}")
        return wrist_row

    if wrist_row is None:
        return external_row

    if tuple(wrist_row.shape) != expected:
        raise ValueError(f"Unexpected wrist row shape {tuple(wrist_row.shape)}; expected {expected}")

    separator = torch.zeros((3, PADDING, COMPOSITE_WIDTH), dtype=torch.uint8)
    return torch.cat([external_row, separator, wrist_row], dim=1)


def create_composite_frame(
    *,
    first_external_frame: Tensor | None = None,
    previous_external_frame: Tensor | None = None,
    current_external_frame: Tensor | None = None,
    first_wrist_frame: Tensor | None = None,
    previous_wrist_frame: Tensor | None = None,
    current_wrist_frame: Tensor | None = None,
) -> Tensor:
    """Build one external-only, wrist-only, or paired temporal composite."""

    external_frames = (first_external_frame, previous_external_frame, current_external_frame)
    wrist_frames = (first_wrist_frame, previous_wrist_frame, current_wrist_frame)

    for view_name, frames in (("external", external_frames), ("wrist", wrist_frames)):
        if any(frame is not None for frame in frames) and not all(frame is not None for frame in frames):
            raise ValueError(f"SOLE-R1 requires all three {view_name} frames or no {view_name} frames")

    if not any(frame is not None for frame in (*external_frames, *wrist_frames)):
        raise ValueError("SOLE-R1 requires at least one camera view")

    external_row = None
    if all(frame is not None for frame in external_frames):
        assert first_external_frame is not None
        assert previous_external_frame is not None
        assert current_external_frame is not None
        external_row = compose_temporal_frames(
            first_external_frame,
            previous_external_frame,
            current_external_frame,
        )

    wrist_row = None
    if all(frame is not None for frame in wrist_frames):
        assert first_wrist_frame is not None
        assert previous_wrist_frame is not None
        assert current_wrist_frame is not None
        wrist_row = compose_temporal_frames(first_wrist_frame, previous_wrist_frame, current_wrist_frame)
    return compose_camera_views(external_row=external_row, wrist_row=wrist_row)


def _composite_to_smart_resized_pil(
    composite: Tensor,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    """Apply the public server's post-composite ``smart_resize`` adapter."""

    from qwen_vl_utils import smart_resize

    array = composite.permute(1, 2, 0).contiguous().cpu().numpy()
    image = Image.fromarray(array, mode="RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


@dataclass
@ProcessorStepRegistry.register(name="soler1_composite")
class SOLER1CompositeProcessorStep(ProcessorStep):
    """Validate ``(B,T,C,H,W)`` trajectories and construct every composite.

    No temporal sampling occurs here. The sequence received from the caller is
    the sequence processed by SOLE-R1.
    """

    external_image_key: str | None = "observation.images.top"
    wrist_image_key: str | None = None

    def reset(self) -> None:
        pass

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError("SOLE-R1 preprocessing requires an observation dictionary")
        if self.external_image_key is None and self.wrist_image_key is None:
            raise ValueError("SOLE-R1 requires at least one camera view")

        external: Tensor | None = None
        if self.external_image_key is not None:
            if self.external_image_key not in observation:
                raise KeyError(f"SOLE-R1 expected external image key {self.external_image_key!r}")
            external = _to_btchw_uint8(
                observation[self.external_image_key], image_key=self.external_image_key
            )

        wrist: Tensor | None = None
        if self.wrist_image_key is not None:
            if self.wrist_image_key not in observation:
                raise KeyError(f"SOLE-R1 expected wrist image key {self.wrist_image_key!r}")
            wrist = _to_btchw_uint8(observation[self.wrist_image_key], image_key=self.wrist_image_key)
        if external is not None and wrist is not None and external.shape[:2] != wrist.shape[:2]:
            raise ValueError(
                "SOLE-R1 received different external and wrist batch/time dimensions: "
                f"{tuple(external.shape[:2])} and {tuple(wrist.shape[:2])}"
            )

        reference = external if external is not None else wrist
        assert reference is not None
        batch_size, trajectory_length = reference.shape[:2]
        batches: list[Tensor] = []
        for batch_index in range(batch_size):
            trajectory: list[Tensor] = []
            for timestep in range(trajectory_length):
                previous = max(timestep - 1, 0)
                trajectory.append(
                    create_composite_frame(
                        first_external_frame=None if external is None else external[batch_index, 0],
                        previous_external_frame=(
                            None if external is None else external[batch_index, previous]
                        ),
                        current_external_frame=(
                            None if external is None else external[batch_index, timestep]
                        ),
                        first_wrist_frame=None if wrist is None else wrist[batch_index, 0],
                        previous_wrist_frame=None if wrist is None else wrist[batch_index, previous],
                        current_wrist_frame=None if wrist is None else wrist[batch_index, timestep],
                    )
                )
            batches.append(torch.stack(trajectory))

        new_observation = dict(observation)
        new_observation[SOLER1_COMPOSITE_IMAGE_KEY] = torch.stack(batches)
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "external_image_key": self.external_image_key,
            "wrist_image_key": self.wrist_image_key,
        }


@dataclass
@ProcessorStepRegistry.register(name="soler1_image_encoder")
class SOLER1ImageEncoderProcessorStep(ProcessorStep):
    """Run smart-resize and Qwen image preprocessing once for all timesteps."""

    model_name: str = "Philip-MIT/SOLE-R1-8B"
    smart_resize_factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 12845056
    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_package("transformers", extra="soler1")
        require_package("qwen-vl-utils", extra="soler1", import_name="qwen_vl_utils")
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError("SOLE-R1 image encoding requires an observation dictionary")
        composites = observation.get(SOLER1_COMPOSITE_IMAGE_KEY)
        if not isinstance(composites, Tensor) or composites.ndim != 5:
            shape = None if composites is None else tuple(torch.as_tensor(composites).shape)
            raise ValueError(f"SOLE-R1 expected composites with shape (B,T,C,H,W); got {shape}")

        batch_size, trajectory_length = composites.shape[:2]
        images = [
            _composite_to_smart_resized_pil(
                composites[batch_index, timestep],
                factor=self.smart_resize_factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            for batch_index in range(batch_size)
            for timestep in range(trajectory_length)
        ]
        encoded = self._processor.image_processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        image_grid_thw = encoded["image_grid_thw"].to(torch.long)
        patch_counts = image_grid_thw.prod(dim=-1)
        if not torch.all(patch_counts == patch_counts[0]):
            raise ValueError("SOLE-R1 expected all composites in a batch to produce the same image grid")

        patches_per_image = int(patch_counts[0].item())
        expected_patches = batch_size * trajectory_length * patches_per_image
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                f"SOLE-R1 image processor returned {pixel_values.shape[0]} patches; expected {expected_patches}"
            )
        merge_size = int(self._processor.image_processor.merge_size)
        image_token_count = patch_counts // (merge_size**2)

        new_observation = dict(observation)
        new_observation.pop(SOLER1_COMPOSITE_IMAGE_KEY)
        new_observation[SOLER1_PIXEL_VALUES_KEY] = pixel_values.reshape(
            batch_size, trajectory_length, patches_per_image, *pixel_values.shape[1:]
        )
        new_observation[SOLER1_IMAGE_GRID_THW_KEY] = image_grid_thw.reshape(batch_size, trajectory_length, 3)
        new_observation[SOLER1_IMAGE_TOKEN_COUNT_KEY] = image_token_count.reshape(
            batch_size, trajectory_length
        )
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "smart_resize_factor": self.smart_resize_factor,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }


def make_soler1_pre_post_processors(
    config: SOLER1Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build serializable static preprocessing and identity postprocessing."""

    del dataset_stats
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            SOLER1CompositeProcessorStep(
                external_image_key=config.external_image_key,
                wrist_image_key=config.wrist_image_key,
            ),
            SOLER1ImageEncoderProcessorStep(
                model_name=config.model_name,
                smart_resize_factor=config.smart_resize_factor,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor
