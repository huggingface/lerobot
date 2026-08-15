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

import builtins
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from torch import Tensor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RYNNVALUE_FEATURE_PREFIX, RynnValueConfig
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from .rynn_value_lang.configuration_rynn_value_lang import RynnValueLangConfig
    from .rynn_value_lang.modeling_rynn_value_lang import RynnValueLangModel
else:
    RynnValueLangConfig = None  # type: ignore[assignment]
    RynnValueLangModel = None  # type: ignore[assignment]

RYNNVALUE_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
)

T = TypeVar("T", bound="RynnValueRewardModel")


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


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
    """Native LeRobot wrapper around official RynnValue checkpoints."""

    name = "rynnvalue"
    config_class = RynnValueConfig

    def __init__(self, config: RynnValueConfig, model: Any | None = None) -> None:
        require_package("transformers", extra="rynnvalue")
        super().__init__(config)
        self.config = config

        if model is not None:
            self.model = model
            return

        dtype = _torch_dtype(config.torch_dtype)
        if config.pretrained_path is not None and config.model_config is None:
            raise ValueError(
                "RynnValue LeRobot checkpoint is missing `model_config`. "
                "Reconvert the official checkpoint with "
                "`python -m lerobot.rewards.rynnvalue.convert_rynnvalue_checkpoint`."
            )
        if config.model_config is not None:
            model_config = RynnValueLangConfig.from_dict(config.model_config)
            model_config._attn_implementation = config.attn_implementation
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                self.model = RynnValueLangModel(model_config)
            finally:
                torch.set_default_dtype(previous_dtype)
            return

        model_config = RynnValueLangConfig.from_pretrained(
            config.model_id,
            revision=config.model_revision,
        )
        model_config._attn_implementation = config.attn_implementation
        self.model = RynnValueLangModel.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            config=model_config,
            dtype=dtype,
        )

    def compute_reward(self, batch: dict[str, Any]) -> Tensor:
        inputs: dict[str, Any] = {}
        for key in RYNNVALUE_MODEL_INPUT_KEYS:
            batch_key = f"{RYNNVALUE_FEATURE_PREFIX}{key}"
            if batch_key in batch:
                inputs[key] = batch[batch_key]
        if "input_ids" not in inputs:
            raise KeyError(
                f"RynnValue batch missing `{RYNNVALUE_FEATURE_PREFIX}input_ids`. "
                "Make sure RynnValueEncoderProcessorStep ran before compute_reward()."
            )
        for required in ("attention_mask", "pixel_values", "image_grid_thw"):
            if required not in inputs:
                raise KeyError(f"RynnValue batch missing `{RYNNVALUE_FEATURE_PREFIX}{required}`.")

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        self.eval()
        with torch.inference_mode():
            outputs = self.model(**inputs)

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
        reward = -remaining_time if self.config.reward_output == "potential" else remaining_time
        return reward.to(self.config.device or "cpu")

    def _save_pretrained(self, save_directory: Path) -> None:
        native_config = getattr(self.model, "config", None)
        if native_config is None or not callable(getattr(native_config, "to_dict", None)):
            raise TypeError("RynnValue native model must expose a serializable `config.to_dict()`")
        self.config.model_config = native_config.to_dict()
        super()._save_pretrained(save_directory)

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
        if not isinstance(config, RynnValueConfig):
            raise TypeError(f"Expected RynnValueConfig, got {type(config).__name__}")
        config.pretrained_path = str(pretrained_name_or_path)
        config.pretrained_revision = revision
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
        )
