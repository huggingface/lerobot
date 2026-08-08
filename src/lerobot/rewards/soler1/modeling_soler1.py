# Copyright 2026 Philip Schroeder, Thomas Weng, Karl Schmeckpeper,
# Eric Rosen, Stephen Hart, Ondrej Biza, and The HuggingFace Inc. team.
# All rights reserved.
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

"""SOLE-R1 inference-only reward model.

Paper:         https://arxiv.org/abs/2603.28730
Project:       https://philip-mit.github.io/sole-r1/
Original code: https://github.com/Philip-MIT/rewardgen
Model:         https://huggingface.co/Philip-MIT/SOLE-R1-8B

SOLE-R1 estimates task progress from a composite image containing the first,
previous, and current robot observations. When two cameras are configured, the
external-camera temporal row is placed above the wrist-camera temporal row.

The preprocessor creates the composite image and stores it under
``observation.soler1.composite_image``. This class builds the SOLE-R1 prompt,
runs the underlying vision-language model with Transformers, parses the
predicted progress percentage, and retains that prediction for the next
timestep's prompt.

This implementation is inference-only and does not depend on RewardGen or
vLLM.
"""

from __future__ import annotations

import builtins
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.processor_soler1 import (
    COMPOSITE_WIDTH,
    SINGLE_VIEW_COMPOSITE_HEIGHT,
    SOLER1_COMPOSITE_IMAGE_KEY,
    TWO_VIEW_COMPOSITE_HEIGHT,
)
from lerobot.utils.import_utils import (
    _transformers_available,
    require_package,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModelForImageTextToText, AutoProcessor
else:
    AutoModelForImageTextToText = None  # type: ignore[assignment, misc]
    AutoProcessor = None  # type: ignore[assignment, misc]


logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SOLER1RewardModel")


SYSTEM_PROMPT = (
    "You are an expert roboticist with the goal of predicting task progress "
    "percentages given frames from a video of a robot attempting to complete "
    "a task. "
    "You first think, in the form of an internal monologue, before providing "
    "your final answer. "
    "Your reasoning process MUST BE enclosed within <think></think> tags and "
    "should include detailed reasoning. "
    "Your final answer MUST BE enclosed within <answer></answer> tags and "
    "should be an integer, positive or negative, representing the current "
    "task progress percentage. "
    "Example output format: "
    "<think>[detailed reasoning process]</think>"
    "<answer>[current task progress]%</answer>"
)

EXTERNAL_AND_WRIST_PROMPT = (
    "Here is an image containing multiple camera views of a robot attempting "
    "to complete a task. "
    "The views on the top are from an external camera. "
    "The views on the bottom are from the robot's wrist camera. "
    "The views from the very first timestep are shown to the left. "
    "The views from the previous timestep are shown in the middle. "
    "The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. "
    "The task progress for the previous timestep is {previous_progress}%. "
    "Predict the task progress for the current timestep."
)

EXTERNAL_ONLY_PROMPT = (
    "Here is an image containing multiple camera views of a robot attempting "
    "to complete a task. "
    "The views are from an external camera. "
    "The views from the very first timestep are shown to the left. "
    "The views from the previous timestep are shown in the middle. "
    "The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. "
    "The task progress for the previous timestep is {previous_progress}%. "
    "Predict the task progress for the current timestep."
)

_ANSWER_PATTERN = re.compile(
    r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*%?\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)

_PERCENT_PATTERN = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*%",
    flags=re.IGNORECASE,
)

_THINK_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _torch_dtype(name: str) -> torch.dtype | str:
    """Resolve a configured torch dtype name."""

    if name == "auto":
        return "auto"

    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype

    raise ValueError(f"Unknown torch dtype: {name!r}")


def _extract_task_value(
    batch: dict[str, Any],
    *,
    task_key: str,
) -> Any:
    """Read the task from the flattened batch or complementary data."""

    if task_key in batch:
        return batch[task_key]

    complementary_data = batch.get("complementary_data")
    if isinstance(complementary_data, dict):
        return complementary_data.get(task_key)

    return None


def _expand_tasks(
    task: Any,
    *,
    batch_size: int,
    default_task: str | None,
) -> list[str]:
    """Expand one task description to match the image batch."""

    if task is None:
        task = default_task

    if task is None:
        raise KeyError("SOLE-R1 expected a task description in complementary data")

    if isinstance(task, str):
        return [task] * batch_size

    if isinstance(task, tuple):
        task = list(task)

    if not (isinstance(task, list) and all(isinstance(item, str) for item in task)):
        raise TypeError(f"SOLE-R1 task must be a string or list of strings; got {type(task).__name__}")

    if len(task) == 1 and batch_size > 1:
        return task * batch_size

    if len(task) != batch_size:
        raise ValueError(f"SOLE-R1 expected {batch_size} task descriptions; got {len(task)}")

    return task


def _composite_batch_to_pil(composites: Tensor) -> list[Image.Image]:
    """Convert ``(B, 3, H, W)`` composite tensors to PIL images."""

    if composites.ndim != 4:
        raise ValueError(
            f"SOLE-R1 expected composite images with shape (B,C,H,W); got {tuple(composites.shape)}"
        )

    if composites.shape[1] != 3:
        raise ValueError(
            f"SOLE-R1 expected RGB composite images with 3 channels; got shape {tuple(composites.shape)}"
        )

    images = composites.detach().cpu()

    if images.is_floating_point():
        images = images.float()
        if images.numel() > 0 and images.max().item() <= 1.0:
            images = images * 255.0

    images = images.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    return [Image.fromarray(image.numpy(), mode="RGB") for image in images]


def parse_progress_percentage(
    completion: str,
    *,
    fallback: float,
) -> float:
    """Parse and clamp the progress percentage produced by SOLE-R1.

    The preferred format is ``<answer>42%</answer>``. If those tags are
    missing, the final percentage appearing in the completion is used.
    If no percentage can be parsed, the previous progress is returned.
    """

    answer_matches = list(_ANSWER_PATTERN.finditer(completion))

    if answer_matches:
        value_text = answer_matches[-1].group(1)
    else:
        percentage_matches = list(_PERCENT_PATTERN.finditer(completion))
        if not percentage_matches:
            logger.warning(
                "Could not parse SOLE-R1 progress from completion; reusing previous progress. Completion: %r",
                completion,
            )
            return fallback

        value_text = percentage_matches[-1].group(1)

    try:
        value = float(value_text)
    except ValueError:
        logger.warning(
            "Could not convert SOLE-R1 progress value %r to float; reusing previous progress.",
            value_text,
        )
        return fallback

    return min(100.0, max(0.0, value))


def extract_reasoning_trace(completion: str) -> str:
    """Extract the optional reasoning trace from a completion."""

    match = _THINK_PATTERN.search(completion)
    if match is None:
        return ""

    return match.group(1).strip()


class SOLER1RewardModel(PreTrainedRewardModel):
    """SOLE-R1 vision-language reward model for inference."""

    name = "sole-r1"
    config_class = SOLER1Config

    def __init__(self, config: SOLER1Config) -> None:
        require_package("transformers", extra="soler1")

        super().__init__(config)
        self.config = config

        torch_dtype = _torch_dtype(config.torch_dtype)

        model_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            **model_kwargs,
        )

        self._previous_progress: list[float] | None = None
        self.last_completions: list[str] = []
        self.last_reasoning_traces: list[str] = []

    def reset(self) -> None:
        """Clear all episode-local progress and completion state."""

        self._previous_progress = None
        self.last_completions = []
        self.last_reasoning_traces = []

    def _validate_composite_shape(self, composites: Tensor) -> None:
        """Validate the fixed dimensions emitted by the preprocessor."""

        if composites.ndim != 4:
            raise ValueError(
                f"SOLE-R1 expected composite images with shape (B,C,H,W); got {tuple(composites.shape)}"
            )

        expected_height = (
            TWO_VIEW_COMPOSITE_HEIGHT
            if self.config.wrist_image_key is not None
            else SINGLE_VIEW_COMPOSITE_HEIGHT
        )

        expected_shape = (
            composites.shape[0],
            3,
            expected_height,
            COMPOSITE_WIDTH,
        )

        if tuple(composites.shape) != expected_shape:
            raise ValueError(
                "SOLE-R1 received a composite image with the wrong shape. "
                f"Got {tuple(composites.shape)}; expected {expected_shape}. "
                "Make sure SOLER1CompositeProcessorStep ran before "
                "compute_reward()."
            )

    def _build_prompt(
        self,
        *,
        task_description: str,
        previous_progress: float,
    ) -> str:
        """Build the user prompt for one sample."""

        template = (
            EXTERNAL_AND_WRIST_PROMPT if self.config.wrist_image_key is not None else EXTERNAL_ONLY_PROMPT
        )

        previous_progress_text = (
            str(int(previous_progress)) if previous_progress.is_integer() else f"{previous_progress:g}"
        )

        return template.format(
            task_description=task_description,
            previous_progress=previous_progress_text,
        )

    def _encode_batch(
        self,
        *,
        images: list[Image.Image],
        tasks: list[str],
        previous_progress: list[float],
    ) -> dict[str, Tensor]:
        """Build and encode the multimodal chat prompts."""

        prompt_texts: list[str] = []

        for image, task, progress in zip(
            images,
            tasks,
            previous_progress,
            strict=True,
        ):
            user_prompt = self._build_prompt(
                task_description=task,
                previous_progress=progress,
            )

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]

            prompt_texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        encoded = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        return dict(encoded)

    def _generation_kwargs(self) -> dict[str, Any]:
        """Build Transformers generation arguments from the configuration."""

        do_sample = self.config.temperature > 0

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                }
            )

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        return generation_kwargs

    @torch.no_grad()
    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """Predict one task-progress percentage per sample.

        Returns a tensor with shape ``(batch_size,)``. Values are percentages
        clamped to ``[0, 100]``.
        """

        if SOLER1_COMPOSITE_IMAGE_KEY not in batch:
            raise KeyError(
                f"SOLE-R1 batch is missing "
                f"{SOLER1_COMPOSITE_IMAGE_KEY!r}. Make sure "
                "SOLER1CompositeProcessorStep ran before compute_reward()."
            )

        composites = batch[SOLER1_COMPOSITE_IMAGE_KEY]

        if not isinstance(composites, Tensor):
            composites = torch.as_tensor(composites)

        self._validate_composite_shape(composites)

        batch_size = composites.shape[0]

        if self._previous_progress is None:
            previous_progress = [0.0] * batch_size
        else:
            if len(self._previous_progress) != batch_size:
                raise ValueError(
                    "SOLE-R1 batch size changed within an episode. Call "
                    "reward_model.reset() before starting a new episode."
                )

            previous_progress = self._previous_progress.copy()

        task_value = _extract_task_value(
            batch,
            task_key=self.config.task_key,
        )
        tasks = _expand_tasks(
            task_value,
            batch_size=batch_size,
            default_task=self.config.default_task,
        )

        images = _composite_batch_to_pil(composites)

        encoded = self._encode_batch(
            images=images,
            tasks=tasks,
            previous_progress=previous_progress,
        )

        model_device = next(self.model.parameters()).device
        encoded = {
            key: value.to(model_device) if isinstance(value, Tensor) else value
            for key, value in encoded.items()
        }

        input_length = encoded["input_ids"].shape[1]

        self.model.eval()
        generated_ids = self.model.generate(
            **encoded,
            **self._generation_kwargs(),
        )

        generated_only_ids = generated_ids[:, input_length:]

        completions = self.processor.batch_decode(
            generated_only_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        progress_values = [
            parse_progress_percentage(
                completion,
                fallback=previous_value,
            )
            for completion, previous_value in zip(
                completions,
                previous_progress,
                strict=True,
            )
        ]

        self._previous_progress = progress_values
        self.last_completions = list(completions)
        self.last_reasoning_traces = [extract_reasoning_trace(completion) for completion in completions]

        return torch.tensor(
            progress_values,
            dtype=torch.float32,
            device=self.config.device or model_device,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save only the LeRobot SOLE-R1 configuration.

        The underlying VLM remains identified by ``config.model_name`` and is
        loaded from its original Hugging Face repository.
        """

        self.config._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: RewardModelConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> T:
        """Load a LeRobot SOLE-R1 configuration and its referenced VLM."""

        del strict

        if config is None:
            config = RewardModelConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        if not isinstance(config, SOLER1Config):
            raise TypeError(
                f"Expected SOLER1Config, got {type(config).__name__}. "
                f"Make sure {pretrained_name_or_path!r} points to a "
                "LeRobot SOLE-R1 configuration."
            )

        model_id = str(pretrained_name_or_path)

        if not os.path.isdir(model_id):
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as error:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} was not found in the Hugging Face repository {model_id!r}"
                ) from error

        instance = cls(config)
        instance.to(config.device)
        instance.eval()
        return instance
