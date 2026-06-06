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

"""Robometer pre/post processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    policy_action_to_transition,
)
from lerobot.rewards.robometer.configuration_robometer import (
    ROBOMETER_SPECIAL_TOKENS,
    RobometerConfig,
)
from lerobot.rewards.robometer.modeling_robometer import ROBOMETER_FEATURE_PREFIX
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

PROGRESS_PROMPT = (
    "The task for the robot is '{task}'. Given the trajectory video, predict "
    "the task progress at each frame, how far along the robot is towards "
    "completing the task, a float between 0 and 1, where 0 is the starting "
    "state and 1 is when the task is completed. If the robot is not "
    "performing the same task, predict 0 progress."
)


def _frames_to_pil(frames: np.ndarray) -> list[Image.Image]:
    """Convert ``(T, H, W, C)`` uint8 frames to a list of PIL images."""
    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C) frames; got shape {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return [Image.fromarray(frames[i]) for i in range(frames.shape[0])]


def _video_to_numpy(video: Tensor, *, max_frames: int | None) -> np.ndarray:
    """Convert one trajectory tensor to a ``(T, H, W, C) uint8`` numpy array."""
    if max_frames is not None:
        video = video[-max_frames:]
    if video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Expected channel dim of size 1 or 3, got shape {tuple(video.shape)}")

    array = video.detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _expand_tasks(task: Any, *, batch_size: int, default: str | None) -> list[str]:
    if task is None:
        task = default
    if task is None:
        raise KeyError("Robometer expected a task description in complementary data")
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, tuple):
        task = list(task)
    if not (isinstance(task, list) and all(isinstance(item, str) for item in task)):
        raise TypeError(f"Robometer task must be a string or list of strings, got {type(task)}")
    if len(task) == 1 and batch_size > 1:
        return task * batch_size
    if len(task) != batch_size:
        raise ValueError(f"Expected {batch_size} tasks, got {len(task)}")
    return task


@dataclass
@ProcessorStepRegistry.register(name="robometer_encoder")
class RobometerEncoderProcessorStep(ProcessorStep):
    """Encode raw frames + task into Qwen-VL tensors for the Robometer model.

    Loads a :class:`~transformers.AutoProcessor` matching ``base_model_id`` and
    registers Robometer's special tokens on the tokenizer. The matching
    embedding resize happens model-side in
    :meth:`RobometerRewardModel.__init__`.

    At call time the step reads:

    - ``observation[image_key]``: ``(B, T, C, H, W)`` or ``(B, C, H, W)`` frames.
    - ``complementary_data[task_key]``: a string or list of strings.

    and writes ``observation[f"{ROBOMETER_FEATURE_PREFIX}<name>"]`` for:

    - the Qwen-VL processor outputs: ``input_ids``, ``attention_mask``,
      ``pixel_values``, ``image_grid_thw``, ``video_grid_thw``, ...
    - Robometer-specific token ids consumed by the model heads:
      ``prog_token_id``, ``vision_start_token_id``, ``vision_end_token_id``,
      ``video_merge_size``.
    """

    base_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 8
    use_multi_image: bool = True
    use_per_frame_progress_token: bool = True
    max_length: int = 1024

    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_package("transformers", extra="robometer")
        require_package("qwen-vl-utils", extra="robometer", import_name="qwen_vl_utils")

        self._processor = AutoProcessor.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
            do_sample_frames=False,
            padding_side="right",
        )

        # Register Robometer's special tokens on the tokenizer. The matching
        # embedding resize happens model-side in `RobometerRewardModel.__init__`.
        tokenizer = self._processor.tokenizer
        # Qwen tokenizers may not define a pad token, but batched prompts/videos
        # require padding, so reuse EOS as the padding token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for token in ROBOMETER_SPECIAL_TOKENS:
            if token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": [token]})

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if not isinstance(observation, dict):
            raise ValueError("RobometerEncoderProcessorStep requires an observation dict")

        if self.image_key not in observation:
            raise KeyError(f"Robometer expected image key {self.image_key!r} in observation")

        frames = observation[self.image_key]
        tensor = frames.detach().cpu() if isinstance(frames, Tensor) else torch.as_tensor(frames)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 5:
            raise ValueError(
                f"Expected Robometer frames with shape (B,C,H,W) or (B,T,C,H,W); got {tuple(tensor.shape)}"
            )

        batch_size = tensor.shape[0]
        tasks = _expand_tasks(
            complementary.get(self.task_key, self.default_task),
            batch_size=batch_size,
            default=self.default_task,
        )

        samples = [
            (_video_to_numpy(tensor[i], max_frames=self.max_frames), tasks[i]) for i in range(batch_size)
        ]
        encoded = self.encode_samples(samples)

        new_observation = dict(observation)
        for key, value in encoded.items():
            new_observation[f"{ROBOMETER_FEATURE_PREFIX}{key}"] = value

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def encode_samples(self, samples: list[tuple[np.ndarray, str]]) -> dict[str, Tensor]:
        """Run the Qwen-VL processor on a list of ``(frames, task)`` samples."""
        from qwen_vl_utils import process_vision_info

        conversations = [self._build_conversation(frames, task) for frames, task in samples]

        texts = [
            self._processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=True,
                enable_thinking=False,
                fps=1,
            )
            for msg in conversations
        ]

        process_kwargs: dict[str, Any] = {
            "return_video_kwargs": True,
            "return_video_metadata": True,
        }
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "patch_size"):
            process_kwargs["image_patch_size"] = image_processor.patch_size

        image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, **process_kwargs)

        videos: list[Any] | None = None
        video_metadatas: list[Any] | None = None
        if video_inputs:
            if isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
                videos_seq, metadatas_seq = zip(*video_inputs, strict=False)
                videos = list(videos_seq)
                video_metadatas = list(metadatas_seq)
            else:
                videos = list(video_inputs)

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "padding": True,
            "truncation": False,
            "max_length": self.max_length,
            "return_tensors": "pt",
            "do_resize": False,
        }
        if videos is not None:
            processor_kwargs["videos"] = videos
        if video_metadatas is not None:
            processor_kwargs["video_metadata"] = video_metadatas
        if video_kwargs:
            processor_kwargs.update(video_kwargs)

        encoded = self._processor(**processor_kwargs)

        # Write Robometer-specific token ids and the video patch merge size into
        # the encoded batch so `RobometerRewardModel` doesn't need its own
        # tokenizer at inference (EO1-style separation: the processor owns the
        # tokenizer, the model owns the backbone and heads).
        tokenizer = self._processor.tokenizer
        encoded["prog_token_id"] = tokenizer.convert_tokens_to_ids("<|prog_token|>")
        encoded["vision_start_token_id"] = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        encoded["vision_end_token_id"] = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        video_processor = getattr(self._processor, "video_processor", None)
        encoded["video_merge_size"] = int(getattr(video_processor, "merge_size", 14))
        return encoded

    def _build_conversation(self, frames: np.ndarray, task: str) -> list[dict[str, Any]]:
        pil_frames = _frames_to_pil(frames)
        prompt = PROGRESS_PROMPT.format(task=task)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if self.use_multi_image:
            for image in pil_frames:
                content.append({"type": "image", "image": image})
                if self.use_per_frame_progress_token:
                    content.append({"type": "text", "text": "<|prog_token|>"})
        else:
            content.append({"type": "video", "video": pil_frames, "sample_fps": 1.0})

        return [{"role": "user", "content": content}]

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "image_key": self.image_key,
            "task_key": self.task_key,
            "default_task": self.default_task,
            "max_frames": self.max_frames,
            "use_multi_image": self.use_multi_image,
            "use_per_frame_progress_token": self.use_per_frame_progress_token,
            "max_length": self.max_length,
        }


def make_robometer_pre_post_processors(
    config: RobometerConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pipeline that pre-encodes frames + task into Qwen-VL tensors.

    The preprocessor adds a batch dimension if needed, runs Robometer's
    encoder, and moves everything to the configured device. The
    postprocessor is the identity since Robometer outputs a single reward
    tensor.
    """
    del dataset_stats  # Robometer has its own normalisation inside the Qwen-VL processor.

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            RobometerEncoderProcessorStep(
                base_model_id=config.base_model_id,
                image_key=config.image_key,
                task_key=config.task_key,
                default_task=config.default_task,
                max_frames=config.max_frames,
                use_multi_image=config.use_multi_image,
                use_per_frame_progress_token=config.use_per_frame_progress_token,
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
