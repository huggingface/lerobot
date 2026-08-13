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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
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
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RYNNVALUE_FEATURE_PREFIX, RynnValueConfig
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from .rynn_value_lang.processing_rynn_value_lang import RynnValueLangProcessor
else:
    RynnValueLangProcessor = None  # type: ignore[assignment]


def _expand_text(value: Any, *, batch_size: int, default: str | None, name: str) -> list[str]:
    if value is None:
        value = default
    if value is None:
        raise KeyError(f"RynnValue expected {name} in complementary data")
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, tuple):
        value = list(value)
    if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
        raise TypeError(f"RynnValue {name} must be a string or list of strings")
    if len(value) == 1 and batch_size > 1:
        return value * batch_size
    if len(value) != batch_size:
        raise ValueError(f"Expected {batch_size} {name} values, got {len(value)}")
    return value


def _uniform_subsample(video: Tensor, max_frames: int | None) -> Tensor:
    if max_frames is None or video.shape[0] <= max_frames:
        return video
    indices = torch.linspace(0, video.shape[0] - 1, max_frames).round().long()
    return video[indices]


def _video_to_pil(video: Tensor, *, max_frames: int | None) -> list[Image.Image]:
    video = _uniform_subsample(video, max_frames)
    if video.ndim != 4:
        raise ValueError(f"Expected a (T,C,H,W) or (T,H,W,C) video, got {tuple(video.shape)}")
    if video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Expected channel dimension of size 1 or 3, got {tuple(video.shape)}")
    array = video.detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating):
        if array.size and array.min() < 0:
            array = (array + 1.0) / 2.0
        if array.size and array.max() <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return [Image.fromarray(frame).convert("RGB") for frame in array]


def _pad_sequences(sequences: list[Tensor], *, padding_value: int) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [sequence.squeeze(0) for sequence in sequences],
        batch_first=True,
        padding_value=padding_value,
    )


@dataclass
@ProcessorStepRegistry.register(name="rynnvalue_encoder")
class RynnValueEncoderProcessorStep(ProcessorStep):
    """Encode trajectory frames and task text with the native RynnValue processor."""

    model_id: str = "Alibaba-DAMO-Academy/RynnValue-4B"
    model_revision: str | None = None
    image_key: str = "observation.images.top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 8
    robot_description: str | None = None
    camera_description: str | None = None
    use_meta: bool | None = None
    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_package("transformers", extra="rynnvalue")
        self._processor = RynnValueLangProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        if self.use_meta is not None:
            self._processor.use_meta = self.use_meta
            self._processor.refresh_conversation_builder()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if self.image_key not in observation:
            raise KeyError(f"RynnValue expected image key {self.image_key!r} in observation")
        frames = observation[self.image_key]
        tensor = frames.detach().cpu() if isinstance(frames, Tensor) else torch.as_tensor(frames)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 5:
            raise ValueError(
                f"Expected RynnValue frames with shape (B,C,H,W) or (B,T,C,H,W); got {tuple(tensor.shape)}"
            )

        tasks = _expand_text(
            complementary.get(self.task_key),
            batch_size=tensor.shape[0],
            default=self.default_task,
            name="task",
        )
        encoded = self.encode_samples(
            [
                (_video_to_pil(tensor[index], max_frames=self.max_frames), tasks[index])
                for index in range(tensor.shape[0])
            ]
        )
        new_observation = dict(observation)
        for key, value in encoded.items():
            new_observation[f"{RYNNVALUE_FEATURE_PREFIX}{key}"] = value
        output = transition.copy()
        output[TransitionKey.OBSERVATION] = new_observation
        return output

    def encode_samples(self, samples: list[tuple[list[Image.Image], str]]) -> dict[str, Tensor]:
        outputs = [
            self._processor.process_episode(
                instruction=task,
                images=images,
                robot_description=self.robot_description,
                camera_description=self.camera_description,
            )
            for images, task in samples
        ]
        pad_id = self._processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._processor.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("RynnValue tokenizer must define a pad or EOS token")

        encoded = {
            "input_ids": _pad_sequences([output["input_ids"] for output in outputs], padding_value=pad_id),
            "attention_mask": _pad_sequences(
                [output["attention_mask"] for output in outputs], padding_value=0
            ),
            "pixel_values": torch.cat([output["pixel_values"].flatten(0, 1) for output in outputs], dim=0),
            "image_grid_thw": torch.cat(
                [output["image_grid_thw"].flatten(0, 1) for output in outputs], dim=0
            ),
        }
        if all("mm_token_type_ids" in output for output in outputs):
            encoded["mm_token_type_ids"] = _pad_sequences(
                [output["mm_token_type_ids"] for output in outputs], padding_value=0
            )
        return encoded

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "image_key": self.image_key,
            "task_key": self.task_key,
            "default_task": self.default_task,
            "max_frames": self.max_frames,
            "robot_description": self.robot_description,
            "camera_description": self.camera_description,
            "use_meta": self.use_meta,
        }


def make_rynnvalue_pre_post_processors(
    config: RynnValueConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    del dataset_stats
    processor_id = str(config.pretrained_path) if config.pretrained_path is not None else config.model_id
    processor_revision = (
        config.pretrained_revision if config.pretrained_path is not None else config.model_revision
    )
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            RynnValueEncoderProcessorStep(
                model_id=processor_id,
                model_revision=processor_revision,
                image_key=config.image_key,
                task_key=config.task_key,
                default_task=config.default_task,
                max_frames=config.max_frames,
                robot_description=config.robot_description,
                camera_description=config.camera_description,
                use_meta=config.use_meta,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor
