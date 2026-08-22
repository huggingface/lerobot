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

"""SOLE-R1 inference-only reward model.

Paper:         https://arxiv.org/abs/2603.28730
Project:       https://philip-mit.github.io/sole-r1/
Model:         https://huggingface.co/Philip-MIT/SOLE-R1-8B

SOLE-R1 estimates task progress from a composite image containing the first,
previous, and current observations. With two cameras, the external-camera
temporal row is placed above the wrist-camera temporal row.

This implementation is inference-only and depends on Transformers rather than
RewardGen or vLLM.
"""

from __future__ import annotations

import builtins
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
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
    SOLER1_ORIGINAL_LENGTH_KEY,
    SOLER1_SAMPLE_INDICES_KEY,
    TWO_VIEW_COMPOSITE_HEIGHT,
)
from lerobot.utils.import_utils import (
    _transformers_available,
    require_package,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )
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
    "should be a integer (positive or negative) representing current "
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
    "The views from the very first timestep are shown to the left. "
    "The views from the previous timestep are shown in the middle. "
    "The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. "
    "The task progress for the previous timestep is {previous_progress}%. "
    "Predict the task progress for the current timestep."
)

WRIST_ONLY_PROMPT = (
    "Here is an image containing multiple camera views of a robot attempting "
    "to complete a task. "
    "The views are from the robot's wrist camera. "
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
    """Read the task from a flattened batch or complementary data."""

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
    """Expand task descriptions to match the trajectory batch."""

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


def _composite_batch_to_pil(
    composites: Tensor,
) -> list[Image.Image]:
    """Convert ``(B,3,H,W)`` composite tensors to PIL images."""

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

    return [
        Image.fromarray(
            image.numpy(),
            mode="RGB",
        )
        for image in images
    ]


def _parse_progress(
    completion: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    """Parse and clamp a SOLE-R1 ``<answer>progress%</answer>`` value."""
    answer_matches = list(_ANSWER_PATTERN.finditer(completion))
    if not answer_matches:
        logger.warning(
            "Could not parse a tagged SOLE-R1 answer from completion: %r",
            completion,
        )
        return None

    try:
        value = float(answer_matches[-1].group(1))
    except ValueError:
        logger.warning(
            "Could not convert SOLE-R1 progress value to float: %r",
            answer_matches[-1].group(1),
        )
        return None

    return min(maximum, max(minimum, value))


def extract_reasoning_trace(
    completion: str,
) -> str:
    """Extract the optional reasoning trace from a completion."""

    match = _THINK_PATTERN.search(completion)
    if match is None:
        return ""

    return match.group(1).strip()


class SOLER1RewardModel(PreTrainedRewardModel):
    """SOLE-R1 vision-language reward model for inference."""

    name = "sole-r1"
    config_class = SOLER1Config

    def __init__(
        self,
        config: SOLER1Config,
    ) -> None:
        require_package(
            "transformers",
            extra="soler1",
        )

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
        """Clear cached diagnostic output from the previous call."""

        self._previous_progress = None
        self.last_completions = []
        self.last_reasoning_traces = []

    def _validate_composite_shape(
        self,
        composites: Tensor,
    ) -> None:
        """Validate composites emitted by the SOLE-R1 preprocessor."""

        if composites.ndim != 5:
            raise ValueError(
                "SOLE-R1 expected trajectory composites with shape "
                f"(B,T,C,H,W); got {tuple(composites.shape)}"
            )

        if composites.shape[1] < 1:
            raise ValueError("SOLE-R1 requires at least one timestep per trajectory")

        has_two_views = self.config.external_image_key is not None and self.config.wrist_image_key is not None
        expected_height = TWO_VIEW_COMPOSITE_HEIGHT if has_two_views else SINGLE_VIEW_COMPOSITE_HEIGHT
        expected_suffix = (
            3,
            expected_height,
            COMPOSITE_WIDTH,
        )

        if tuple(composites.shape[2:]) != expected_suffix:
            raise ValueError(
                "SOLE-R1 received trajectory composites with the "
                f"wrong shape. Got {tuple(composites.shape)}; "
                f"expected (B,T,{expected_suffix[0]},"
                f"{expected_suffix[1]},{expected_suffix[2]}). "
                "Make sure SOLER1CompositeProcessorStep ran before "
                "compute_reward()."
            )

    def _build_prompt(
        self,
        *,
        task_description: str,
        previous_progress: float,
    ) -> str:
        """Build the user prompt for one trajectory."""

        if self.config.external_image_key is not None and self.config.wrist_image_key is not None:
            template = EXTERNAL_AND_WRIST_PROMPT
        elif self.config.wrist_image_key is not None:
            template = WRIST_ONLY_PROMPT
        else:
            template = EXTERNAL_ONLY_PROMPT

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
        """Build and encode a batch of multimodal prompts."""

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
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_length = encoded["input_ids"].shape[1]
        if input_length > self.config.max_input_length:
            raise ValueError(
                f"SOLE-R1 input length {input_length} exceeds "
                f"max_input_length {self.config.max_input_length}."
            )

        return dict(encoded)

    def _generation_kwargs(
        self,
    ) -> dict[str, Any]:
        """Build Transformers generation arguments."""

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

        tokenizer = getattr(
            self.processor,
            "tokenizer",
            None,
        )

        if tokenizer is not None:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        return generation_kwargs

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """Return one final reward per trajectory.

        Returns:
            Tensor of shape ``(B,)``.
        """
        return self._compute_rewards(batch, dense=False)

    def compute_progress(self, batch: dict[str, Any]) -> Tensor:
        """Return scaled progress for every original trajectory timestep.

        Sampled anchor predictions are linearly interpolated back to the original
        trajectory length.

        Returns:
            Tensor of shape ``(B, T)``.
        """
        if self.config.reward_output != "progress":
            raise ValueError(
                "compute_progress() requires reward_output='progress'; "
                "use compute_reward() for binary success output."
            )

        return self._compute_rewards(batch, dense=True)

    @torch.no_grad()
    def _compute_rewards(
        self,
        batch: dict[str, Any],
        *,
        dense: bool = False,
    ) -> Tensor:
        """Generate SOLE-R1 progress for one or more batched trajectories.

        The first sampled timestep is assigned 0% progress. Subsequent predictions
        are generated sequentially because each prediction conditions on the
        preceding predicted progress.

        Args:
            batch: Output of ``SOLER1CompositeProcessorStep``. Composite images
                must have shape ``(B, S, C, H, W)``.
            dense: Whether to interpolate sampled progress predictions back to
                every original trajectory timestep.

        Returns:
            A tensor of shape ``(B,)`` when ``dense=False`` or ``(B, T)`` when
            ``dense=True``.
        """

        if SOLER1_COMPOSITE_IMAGE_KEY not in batch:
            raise KeyError(
                "SOLE-R1 batch is missing "
                f"{SOLER1_COMPOSITE_IMAGE_KEY!r}. Make sure "
                "SOLER1CompositeProcessorStep ran before "
                "compute_reward()."
            )

        composites = batch[SOLER1_COMPOSITE_IMAGE_KEY]

        if not isinstance(composites, Tensor):
            composites = torch.as_tensor(composites)

        self._validate_composite_shape(composites)

        batch_size, sampled_length = composites.shape[:2]

        raw_sample_indices = batch.get(SOLER1_SAMPLE_INDICES_KEY)
        raw_original_length = batch.get(SOLER1_ORIGINAL_LENGTH_KEY)

        # Sampling metadata is optional for direct calls using already
        # constructed composites. Without metadata, no interpolation is
        # necessary.
        if raw_sample_indices is None:
            sample_indices = np.arange(
                sampled_length,
                dtype=int,
            )
        else:
            if isinstance(
                raw_sample_indices,
                Tensor,
            ):
                raw_sample_indices = raw_sample_indices.detach().cpu().numpy()

            sample_indices = np.asarray(
                raw_sample_indices,
                dtype=int,
            ).reshape(-1)

        if len(sample_indices) != sampled_length:
            raise ValueError(f"SOLE-R1 expected {sampled_length} sample indices; got {len(sample_indices)}")

        if raw_original_length is None:
            original_length = sampled_length
        elif isinstance(
            raw_original_length,
            Tensor,
        ):
            original_length = int(raw_original_length.detach().cpu().reshape(-1)[0].item())
        else:
            original_length = int(raw_original_length)

        if original_length < 1:
            raise ValueError(f"SOLE-R1 original trajectory length must be positive; got {original_length}")

        if np.any(sample_indices < 0) or np.any(sample_indices >= original_length):
            raise ValueError("SOLE-R1 received sample indices outside the original trajectory range")

        if len(sample_indices) > 1 and np.any(np.diff(sample_indices) <= 0):
            raise ValueError("SOLE-R1 sample indices must be strictly increasing")

        task_value = _extract_task_value(
            batch,
            task_key=self.config.task_key,
        )
        tasks = _expand_tasks(
            task_value,
            batch_size=batch_size,
            default_task=self.config.default_task,
        )

        model_device = next(self.model.parameters()).device

        # The first sampled frame is the original first frame and is
        # defined as exactly 0% progress.
        progress_values = [0.0] * batch_size
        sampled_progress = torch.zeros(
            batch_size,
            sampled_length,
            dtype=torch.float32,
            device=model_device,
        )

        self._previous_progress = progress_values.copy()
        self.last_completions = [""] * batch_size
        self.last_reasoning_traces = [""] * batch_size

        # Sampled positions are evaluated sequentially because position
        # s is conditioned on the prediction at sampled position s - 1.
        self.eval()
        for sampled_position in range(
            1,
            sampled_length,
        ):
            original_timestep = int(sample_indices[sampled_position])

            # Composites are indexed by sampled position, not by their
            # original trajectory timestep.
            images = _composite_batch_to_pil(composites[:, sampled_position])

            encoded = self._encode_batch(
                images=images,
                tasks=tasks,
                previous_progress=progress_values,
            )
            encoded = {
                key: (value.to(model_device) if isinstance(value, Tensor) else value)
                for key, value in encoded.items()
            }

            input_length = encoded["input_ids"].shape[1]

            generated_ids = self.model.generate(
                **encoded,
                **self._generation_kwargs(),
            )
            generated_only_ids = generated_ids[
                :,
                input_length:,
            ]

            completions = self.processor.batch_decode(
                generated_only_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            next_progress_values: list[float] = []

            for completion, previous_value in zip(
                completions,
                progress_values,
                strict=True,
            ):
                progress = _parse_progress(
                    completion,
                    minimum=self.config.min_progress,
                    maximum=self.config.max_progress,
                )

                if progress is None:
                    if not self.config.fallback_to_previous:
                        raise ValueError(
                            "Could not parse SOLE-R1 completion at "
                            f"original timestep {original_timestep}: "
                            f"{completion!r}"
                        )

                    progress = previous_value

                next_progress_values.append(progress)

            progress_values = next_progress_values

            sampled_progress[
                :,
                sampled_position,
            ] = torch.tensor(
                progress_values,
                dtype=torch.float32,
                device=model_device,
            )

            self._previous_progress = progress_values.copy()
            self.last_completions = list(completions)
            self.last_reasoning_traces = [extract_reasoning_trace(completion) for completion in completions]

        progress_percentages = sampled_progress

        if dense and sampled_length != original_length:
            target_indices = np.arange(
                original_length,
                dtype=float,
            )

            interpolated = np.stack(
                [
                    np.interp(
                        target_indices,
                        sample_indices.astype(float),
                        values.detach().cpu().numpy(),
                    )
                    for values in sampled_progress
                ]
            )

            progress_percentages = torch.as_tensor(
                interpolated,
                dtype=torch.float32,
                device=model_device,
            )

        rewards = self._format_rewards(
            progress_percentages,
            dense=dense,
        )

        return rewards

    def _format_rewards(
        self,
        progress_percentages: Tensor,
        *,
        dense: bool,
    ) -> Tensor:
        """Scale progress and select final, dense, or success output."""
        if progress_percentages.ndim != 2:
            raise ValueError(
                "SOLE-R1 expected progress percentages with shape "
                f"(B,T); got {tuple(progress_percentages.shape)}"
            )

        progress_rewards = progress_percentages.float() * self.config.reward_scale
        final_progress = progress_rewards[:, -1]

        if self.config.reward_output == "success":
            rewards = (final_progress > self.config.success_threshold).float()
        elif dense:
            rewards = progress_rewards
        else:
            rewards = final_progress

        return rewards.to(self.config.device or "cpu")

    def _save_pretrained(
        self,
        save_directory: Path,
    ) -> None:
        """Save only the LeRobot SOLE-R1 configuration."""

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
        """Load a SOLE-R1 wrapper configuration and referenced VLM."""

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
                "Expected SOLER1Config, got "
                f"{type(config).__name__}. Make sure "
                f"{pretrained_name_or_path!r} points to a "
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
