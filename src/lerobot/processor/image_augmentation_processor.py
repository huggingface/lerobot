#!/usr/bin/env python

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

"""Train-time image augmentation on the policy device, for `image_transforms.backend="gpu"`.

`lerobot-train` puts this step in a pipeline of its own and applies it to each training batch before the
policy preprocessor, so the workers only decode and the augmentation runs once per batch on the device
the batch is headed to anyway. The policy's own preprocessor is never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation
from lerobot.transforms import BatchedImageTransforms, ImageTransformConfig, ImageTransformsConfig
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


def _config_from_dict(config: dict[str, Any]) -> ImageTransformsConfig:
    tfs = {name: ImageTransformConfig(**tf_cfg) for name, tf_cfg in config.get("tfs", {}).items()}
    return ImageTransformsConfig(**{**config, "tfs": tfs})


@dataclass
@ProcessorStepRegistry.register(name="image_augmentation_processor")
class ImageAugmentationProcessorStep(ObservationProcessorStep):
    """Augment the camera frames of a batch with `BatchedImageTransforms`, independent parameters per sample.

    Args:
        config (`ImageTransformsConfig`, *optional*):
            Which transforms to sample and how; the same configuration the dataloader backend uses.
        image_keys (`list[str]`, *optional*):
            Observation keys to augment. `None` selects every key under `observation.images` and
            `observation.image`.
        device (`str`, *optional*):
            Device to draw the random parameters on and to move the frames to first, e.g. `"cuda"`. `None`
            transforms the frames where they are.
        chunk_size (`int`, *optional*):
            Transform at most this many samples at a time to bound the memory of the intermediate frames.
        compile_model (`bool`, *optional*, defaults to `False`):
            Wrap the transform math in `torch.compile`.
        seed (`int`, *optional*):
            Seed of the step's own generator, which every random draw comes from, so the augmentation is
            reproducible and never consumes the default generators the policy samples from.
    """

    config: ImageTransformsConfig | dict[str, Any] = field(default_factory=ImageTransformsConfig)
    image_keys: list[str] | None = None
    device: str | None = None
    chunk_size: int | None = None
    compile_model: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.config, dict):
            self.config = _config_from_dict(self.config)
        self._transforms = BatchedImageTransforms(
            self.config, chunk_size=self.chunk_size, compile_model=self.compile_model
        )
        self._device = torch.device(self.device) if self.device is not None else None
        # A generator is bound to a device; without one it is built where the first frames arrive.
        self._generator = self._make_generator(self._device) if self._device is not None else None

    def _make_generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        return generator

    def _selected_keys(self, observation: RobotObservation) -> list[str]:
        if self.image_keys is not None:
            return [key for key in self.image_keys if key in observation]
        return [key for key in observation if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}.")]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """Augment the selected image tensors of the observation.

        Args:
            observation (`dict[str, Any]`):
                The observation of the current transition.

        Returns:
            `dict[str, Any]`: The observation with its image tensors augmented, on `device` if one is set.
        """
        for key in self._selected_keys(observation):
            images = observation[key]
            if not isinstance(images, Tensor):
                continue
            if self._device is not None and images.device != self._device:
                images = images.to(self._device, non_blocking=self._device.type == "cuda")
            if self._generator is None:
                self._generator = self._make_generator(images.device)
            observation[key] = self._transforms(images, generator=self._generator)
        return observation

    def get_config(self) -> dict[str, Any]:
        """Return the serializable configuration."""
        assert isinstance(self.config, ImageTransformsConfig)
        return {
            "config": asdict(self.config),
            "image_keys": self.image_keys,
            "device": self.device,
            "chunk_size": self.chunk_size,
            "compile_model": self.compile_model,
            "seed": self.seed,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Augmentation keeps every feature's shape and type."""
        return features
