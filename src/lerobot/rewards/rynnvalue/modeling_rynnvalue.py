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

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.prediction import RemainingTimeOutput
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.rynnvalue.configuration_rynnvalue import (
    RYNNVALUE_FEATURE_PREFIX,
    RYNNVALUE_PROCESSOR_SUBFOLDER,
    RynnValueConfig,
)
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from .rynn_value_lang.configuration_rynn_value_lang import RynnValueLangConfig
    from .rynn_value_lang.modeling_rynn_value_lang import RynnValueLangModel
    from .rynn_value_lang.processing_rynn_value_lang import RynnValueLangProcessor
else:
    RynnValueLangConfig = None  # type: ignore[assignment]
    RynnValueLangModel = None  # type: ignore[assignment]
    RynnValueLangProcessor = None  # type: ignore[assignment]

# The official inference parses these answers from the generated analysis block.
_DESCRIPTION_PATTERN = re.compile(r"-\s*Video Description:\s*(.+)", re.IGNORECASE)
_MATCH_PATTERN = re.compile(r"-\s*Match:\s*(Yes|No)", re.IGNORECASE)
_SUCCESS_PATTERN = re.compile(r"-\s*Success:\s*(Yes|No)", re.IGNORECASE)

RYNNVALUE_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
)


@dataclass(frozen=True)
class RynnValueSuccessOutput:
    """Generated success and task-match answers for each trajectory, shaped `(batch,)`.

    `success` and `match` are 1.0 for "Yes", 0.0 for "No" and NaN when the analysis omits the answer.
    `description` holds each generated video description, or None.
    """

    success: Tensor
    match: Tensor
    description: list[str | None]


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


def _parse_answer(pattern: re.Pattern[str], text: str) -> str | None:
    found = pattern.search(text)
    return found.group(1).strip() if found else None


def _yes_no(answer: str | None) -> float:
    return math.nan if answer is None else float(answer.lower() == "yes")


def _left_pad(inputs: dict[str, Any]) -> dict[str, Any]:
    """Move the preprocessor's right padding to the left, where generation expects it."""
    attention_mask = inputs["attention_mask"]
    shifts = (attention_mask.shape[1] - attention_mask.sum(dim=1)).tolist()
    padded = dict(inputs)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key in padded:
            padded[key] = torch.stack(
                [torch.roll(row, shift) for row, shift in zip(padded[key], shifts, strict=True)]
            )
    return padded


def reduce_remaining_time(
    pred_value: Tensor,
    *,
    batch_size: int,
    slot_counts: Tensor | list[int] | None = None,
) -> Tensor:
    """Average head ensembles and select each sample's final value slot."""
    values = pred_value.float()
    if values.ndim == 1:
        flattened = values
    elif values.ndim == 2:
        # Ensemble output is (num_heads, batch_size * num_slots). A single-head
        # checkpoint uses the same shape with num_heads=1.
        flattened = values.mean(dim=0)
    else:
        raise ValueError(f"Unexpected RynnValue prediction shape: {tuple(values.shape)}")

    if slot_counts is not None:
        counts = torch.as_tensor(slot_counts, device=flattened.device, dtype=torch.long)
        if counts.ndim != 1 or counts.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} RynnValue slot counts, got shape {tuple(counts.shape)}")
        if torch.any(counts < 1):
            raise ValueError(
                f"Each RynnValue sample must contain at least one value slot, got {counts.tolist()}"
            )
        if int(counts.sum()) != flattened.numel():
            raise ValueError(
                f"RynnValue slot counts sum to {int(counts.sum())}, "
                f"but the model returned {flattened.numel()} predictions"
            )
        return flattened[counts.cumsum(dim=0) - 1]

    if flattened.numel() % batch_size:
        raise ValueError(
            f"Cannot split RynnValue output shape {tuple(values.shape)} across batch size {batch_size}"
        )
    return flattened.view(batch_size, -1)[:, -1]


class RynnValueRewardModel(PreTrainedRewardModel):
    """RynnValue zero-shot reward model.

    The model builds the empty architecture from `model_config`, which the LeRobot checkpoint embeds in its
    `config.json`, and `from_pretrained()` fills it from `model.safetensors`. Saving embeds `model_config`
    and the checkpoint's `processor` folder.
    """

    config: RynnValueConfig
    name = "rynnvalue"
    config_class = RynnValueConfig

    def __init__(self, config: RynnValueConfig) -> None:
        require_package("transformers", extra="rynnvalue")
        super().__init__(config)
        self.config = config

        dtype = _torch_dtype(config.torch_dtype)
        if config.model_config is None:
            # A fresh config, such as `--reward_model.type=rynnvalue`, takes its checkpoint's architecture.
            checkpoint_config = RewardModelConfig.from_pretrained(
                str(config.pretrained_path), revision=config.pretrained_revision
            )
            config.model_config = cast(RynnValueConfig, checkpoint_config).model_config
        model_config = RynnValueLangConfig.from_dict(config.model_config)
        model_config._attn_implementation = "pred_slot_isolated_eager"
        # Since transformers 5.22, Qwen3-VL passes the value-slot tensors on to the vision tower
        # too, which the official model runs without them, i.e. as plain eager attention.
        model_config.vision_config._attn_implementation = "eager"
        # The value heads are built in float32; like the official inference, cast the whole model.
        self.model = RynnValueLangModel._from_config(model_config, dtype=dtype).to(dtype=dtype)

    def predict_remaining_time(self, batch: Mapping[str, Any]) -> RemainingTimeOutput:
        """Predict the seconds until the task is completed, at the last frame of each trajectory.

        It changes neither the model mode nor gradients: for inference, call it in `eval()` mode under
        `torch.inference_mode()`.

        Args:
            batch (`Mapping[str, Any]`):
                Output of the RynnValue preprocessor.

        Returns:
            `RemainingTimeOutput`: One remaining time per trajectory.
        """
        inputs = self._model_inputs(batch)
        outputs = self.model(**inputs, use_cache=False)
        pred_value = outputs.value.pred_value
        if pred_value is None:
            raise RuntimeError("RynnValue checkpoint did not produce absolute temporal values")

        slot_counts = None
        value_token_id = getattr(self.model.config, "value_token_id", None)
        value_token_repeat = int(getattr(self.model.config, "value_token_repeat", 1))
        if value_token_id is not None:
            token_counts = inputs["input_ids"].eq(value_token_id).sum(dim=1)
            if torch.any(token_counts.remainder(value_token_repeat)):
                raise RuntimeError(
                    f"RynnValue value-token counts {token_counts.tolist()} are not divisible "
                    f"by value_token_repeat={value_token_repeat}"
                )
            slot_counts = token_counts.div(value_token_repeat, rounding_mode="floor")

        remaining_time = reduce_remaining_time(
            pred_value,
            batch_size=int(inputs["input_ids"].shape[0]),
            slot_counts=slot_counts,
        )
        return RemainingTimeOutput(remaining_time_s=remaining_time)

    def predict_success(
        self, batch: Mapping[str, Any], *, max_new_tokens: int = 128
    ) -> RynnValueSuccessOutput:
        """Generate RynnValue's trajectory analysis and read whether the task succeeded.

        As in the official inference, greedy decoding writes an analysis block after the encoded trajectory:
        a video description, whether the video matches the task, and whether the agent completed it. It
        changes neither the model mode nor gradients: for inference, call it in `eval()` mode under
        `torch.inference_mode()`.

        Args:
            batch (`Mapping[str, Any]`):
                Output of the RynnValue preprocessor.
            max_new_tokens (`int`, *optional*, defaults to 128):
                Maximum number of generated tokens, as in the official inference.

        Returns:
            `RynnValueSuccessOutput`: Success and task match, one per trajectory.
        """
        inputs = _left_pad(self._model_inputs(batch))
        tokenizer = self._processor.tokenizer
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sequences = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        texts = tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)
        device = inputs["input_ids"].device
        return RynnValueSuccessOutput(
            success=torch.tensor(
                [_yes_no(_parse_answer(_SUCCESS_PATTERN, text)) for text in texts], device=device
            ),
            match=torch.tensor(
                [_yes_no(_parse_answer(_MATCH_PATTERN, text)) for text in texts], device=device
            ),
            description=[_parse_answer(_DESCRIPTION_PATTERN, text) for text in texts],
        )

    @cached_property
    def _processor(self) -> Any:
        """The checkpoint's tokenizer and image processor, loaded on first use."""
        return RynnValueLangProcessor.from_pretrained(
            str(self.config.pretrained_path),
            revision=self.config.pretrained_revision,
            subfolder=RYNNVALUE_PROCESSOR_SUBFOLDER,
        )

    def _model_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key in RYNNVALUE_MODEL_INPUT_KEYS:
            batch_key = f"{RYNNVALUE_FEATURE_PREFIX}{key}"
            if batch_key in batch:
                inputs[key] = batch[batch_key]
        if "input_ids" not in inputs:
            raise KeyError(
                f"RynnValue batch missing `{RYNNVALUE_FEATURE_PREFIX}input_ids`. "
                "Make sure RynnValueEncoderProcessorStep ran before calling the model."
            )
        for required in ("attention_mask", "pixel_values", "image_grid_thw"):
            if required not in inputs:
                raise KeyError(f"RynnValue batch missing `{RYNNVALUE_FEATURE_PREFIX}{required}`.")
        return inputs

    def _save_pretrained(self, save_directory: Path) -> None:
        from lerobot.distributed.utils import is_main_process

        self.config.model_config = self.model.config.to_dict()
        super()._save_pretrained(save_directory)
        if is_main_process():
            self._processor.save_pretrained(save_directory / RYNNVALUE_PROCESSOR_SUBFOLDER)
