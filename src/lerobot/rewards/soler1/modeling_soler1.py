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

"""Autoregressive inference for the SOLE-R1 reward model."""

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
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.processor_soler1 import (
    SOLER1_IMAGE_GRID_THW_KEY,
    SOLER1_IMAGE_TOKEN_COUNT_KEY,
    SOLER1_PIXEL_VALUES_KEY,
)
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModelForImageTextToText, AutoTokenizer
else:
    AutoModelForImageTextToText = None
    AutoTokenizer = None

logger = logging.getLogger(__name__)
T = TypeVar("T", bound="SOLER1RewardModel")

SYSTEM_PROMPT = (
    "You are an expert roboticist with the goal of predicting task progress "
    "percentages given frames from a video of a robot attempting to complete a task. "
    "You first think, in the form of an internal monologue, before providing your final answer. "
    "Your reasoning process MUST BE enclosed within <think></think> tags and should include detailed reasoning. "
    "Your final answer MUST BE enclosed within <answer></answer> tags and should be a integer "
    "(positive or negative) representing current task progress percentage. "
    "Example output format: <think>[detailed reasoning process]</think>"
    "<answer>[current task progress]%</answer>"
)

EXTERNAL_AND_WRIST_PROMPT = (
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
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
    "Here is an image containing multiple camera views of a robot attempting to complete a task. "
    "The views from the very first timestep are shown to the left. "
    "The views from the previous timestep are shown in the middle. "
    "The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. "
    "The task progress for the previous timestep is {previous_progress}%. "
    "Predict the task progress for the current timestep."
)

WRIST_ONLY_PROMPT = (
    "Here is an image containing multiple views from the robot's wrist camera while it is attempting "
    "to complete a task. "
    "The views from the very first timestep are shown to the left. "
    "The views from the previous timestep are shown in the middle. "
    "The views from the current timestep are shown to the right. "
    "The task description is: {task_description}. "
    "The task progress for the very first timestep is 0%. "
    "The task progress for the previous timestep is {previous_progress}%. "
    "Predict the task progress for the current timestep."
)

_ANSWER_PATTERN = re.compile(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*%?\s*</answer>", re.IGNORECASE | re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def _torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


def _extract_task_value(batch: dict[str, Any], *, task_key: str) -> Any:
    if task_key in batch:
        return batch[task_key]
    complementary_data = batch.get("complementary_data")
    if isinstance(complementary_data, dict):
        return complementary_data.get(task_key)
    return None


def _expand_tasks(task: Any, *, batch_size: int, default_task: str | None) -> list[str]:
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


def _parse_progress(completion: str, *, minimum: float, maximum: float) -> float | None:
    matches = list(_ANSWER_PATTERN.finditer(completion))
    if not matches:
        logger.warning("Could not parse a tagged SOLE-R1 answer from completion: %r", completion)
        return None
    try:
        value = float(matches[-1].group(1))
    except ValueError:
        logger.warning("Could not convert SOLE-R1 progress value to float: %r", matches[-1].group(1))
        return None
    return min(maximum, max(minimum, value))


def extract_reasoning_trace(completion: str) -> str:
    match = _THINK_PATTERN.search(completion)
    return "" if match is None else match.group(1).strip()


class SOLER1RewardModel(PreTrainedRewardModel):
    """Inference-only SOLE-R1 model consuming statically prepared vision tensors."""

    name = "sole-r1"
    config_class = SOLER1Config

    def __init__(self, config: SOLER1Config) -> None:
        require_package("transformers", extra="soler1")
        super().__init__(config)
        self.config = config

        model_kwargs: dict[str, Any] = {
            "dtype": _torch_dtype(config.torch_dtype),
            "trust_remote_code": True,
        }
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForImageTextToText.from_pretrained(config.model_name, **model_kwargs)

        self._previous_progress: list[float] | None = None
        self.last_completions: list[list[str]] = []
        self.last_reasoning_traces: list[list[str]] = []

    def reset(self) -> None:
        self._previous_progress = None
        self.last_completions = []
        self.last_reasoning_traces = []

    def _build_prompt(self, *, task_description: str, previous_progress: float) -> str:
        has_external = self.config.external_image_key is not None
        has_wrist = self.config.wrist_image_key is not None
        if has_external and has_wrist:
            template = EXTERNAL_AND_WRIST_PROMPT
        elif has_external:
            template = EXTERNAL_ONLY_PROMPT
        elif has_wrist:
            template = WRIST_ONLY_PROMPT
        else:
            raise ValueError("SOLE-R1 requires at least one camera view")
        previous = str(int(previous_progress)) if previous_progress.is_integer() else f"{previous_progress:g}"
        return template.format(task_description=task_description, previous_progress=previous)

    def _tokenize_batch(
        self,
        *,
        tasks: list[str],
        previous_progress: list[float],
        image_token_counts: Tensor,
        device: torch.device,
    ) -> dict[str, Tensor]:
        image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
        prompt_texts: list[str] = []
        for task, progress, token_count in zip(
            tasks, previous_progress, image_token_counts.tolist(), strict=True
        ):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": self._build_prompt(task_description=task, previous_progress=progress),
                        },
                    ],
                },
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if prompt.count(image_token) != 1:
                raise ValueError("SOLE-R1 chat template must contain exactly one image token")
            prompt_texts.append(prompt.replace(image_token, image_token * int(token_count), 1))

        encoded = self.tokenizer(
            prompt_texts,
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.config.max_input_length:
            raise ValueError(
                f"SOLE-R1 input length {encoded['input_ids'].shape[1]} exceeds "
                f"max_input_length {self.config.max_input_length}"
            )
        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        encoded["mm_token_type_ids"] = (encoded["input_ids"] == image_token_id).long()
        return {key: value.to(device) for key, value in encoded.items() if isinstance(value, Tensor)}

    def _generation_kwargs(self) -> dict[str, Any]:
        do_sample = self.config.temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        return kwargs

    def _prepared_inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        missing = [
            key
            for key in (SOLER1_PIXEL_VALUES_KEY, SOLER1_IMAGE_GRID_THW_KEY, SOLER1_IMAGE_TOKEN_COUNT_KEY)
            if key not in batch
        ]
        if missing:
            raise KeyError(
                f"SOLE-R1 batch is missing statically prepared inputs {missing}. "
                "Run make_soler1_pre_post_processors() before inference."
            )
        pixel_values = torch.as_tensor(batch[SOLER1_PIXEL_VALUES_KEY])
        image_grid_thw = torch.as_tensor(batch[SOLER1_IMAGE_GRID_THW_KEY])
        image_token_count = torch.as_tensor(batch[SOLER1_IMAGE_TOKEN_COUNT_KEY])
        if pixel_values.ndim < 4:
            raise ValueError(
                f"SOLE-R1 expected pixel_values with shape (B,T,P,...); got {tuple(pixel_values.shape)}"
            )
        if image_grid_thw.ndim != 3 or image_grid_thw.shape[-1] != 3:
            raise ValueError(
                f"SOLE-R1 expected image_grid_thw with shape (B,T,3); got {tuple(image_grid_thw.shape)}"
            )
        if image_token_count.ndim != 2:
            raise ValueError(
                f"SOLE-R1 expected image_token_count with shape (B,T); got {tuple(image_token_count.shape)}"
            )
        if (
            pixel_values.shape[:2] != image_grid_thw.shape[:2]
            or pixel_values.shape[:2] != image_token_count.shape
        ):
            raise ValueError("SOLE-R1 prepared vision tensors disagree on batch/time dimensions")
        if pixel_values.shape[0] < 1 or pixel_values.shape[1] < 1:
            raise ValueError("SOLE-R1 requires at least one batch element and one timestep")
        return pixel_values, image_grid_thw, image_token_count

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        """Return final progress or success with shape ``(B,)``."""
        return self._compute_rewards(batch, dense=False)

    def compute_progress(self, batch: dict[str, Any]) -> Tensor:
        """Return progress for every supplied timestep with shape ``(B,T)``."""
        if self.config.reward_output != "progress":
            raise ValueError("compute_progress() requires reward_output='progress'")
        return self._compute_rewards(batch, dense=True)

    @torch.no_grad()
    def _compute_rewards(self, batch: dict[str, Any], *, dense: bool) -> Tensor:
        pixel_values, image_grid_thw, image_token_count = self._prepared_inputs(batch)
        batch_size, trajectory_length = pixel_values.shape[:2]
        tasks = _expand_tasks(
            _extract_task_value(batch, task_key=self.config.task_key),
            batch_size=batch_size,
            default_task=self.config.default_task,
        )

        model_device = next(self.model.parameters()).device
        progress_values = [0.0] * batch_size
        progress = torch.zeros(batch_size, trajectory_length, dtype=torch.float32, device=model_device)
        self._previous_progress = progress_values.copy()
        self.last_completions = [[""] * trajectory_length for _ in range(batch_size)]
        self.last_reasoning_traces = [[""] * trajectory_length for _ in range(batch_size)]

        self.eval()
        for timestep in range(1, trajectory_length):
            encoded = self._tokenize_batch(
                tasks=tasks,
                previous_progress=progress_values,
                image_token_counts=image_token_count[:, timestep].detach().cpu(),
                device=model_device,
            )
            input_length = encoded["input_ids"].shape[1]
            timestep_pixels = pixel_values[:, timestep].reshape(-1, *pixel_values.shape[3:])
            generated_ids = self.model.generate(
                **encoded,
                pixel_values=timestep_pixels,
                image_grid_thw=image_grid_thw[:, timestep],
                **self._generation_kwargs(),
            )
            completions = self.tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            next_values: list[float] = []
            for batch_index, (completion, previous_value) in enumerate(
                zip(completions, progress_values, strict=True)
            ):
                value = _parse_progress(
                    completion,
                    minimum=self.config.min_progress,
                    maximum=self.config.max_progress,
                )
                if value is None:
                    if not self.config.fallback_to_previous:
                        raise ValueError(
                            f"Could not parse SOLE-R1 completion at timestep {timestep}: {completion!r}"
                        )
                    value = previous_value
                next_values.append(value)
                self.last_completions[batch_index][timestep] = completion
                self.last_reasoning_traces[batch_index][timestep] = extract_reasoning_trace(completion)

            progress_values = next_values
            progress[:, timestep] = torch.tensor(progress_values, dtype=torch.float32, device=model_device)
            self._previous_progress = progress_values.copy()

        scaled = progress * self.config.reward_scale
        final = scaled[:, -1]
        if self.config.reward_output == "success":
            return (final > self.config.success_threshold).float()
        return scaled if dense else final

    def _save_pretrained(self, save_directory: Path) -> None:
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
            raise TypeError(f"Expected SOLER1Config, got {type(config).__name__}")

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
