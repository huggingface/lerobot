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

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.rewards.prediction import RemainingTimeOutput
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
    """RynnValue zero-shot reward model.

    A config without `model_config` builds the model from the official checkpoint `model_id`. Saving
    embeds `model_config`, so a LeRobot checkpoint rebuilds the architecture from its own
    `config.json` before `from_pretrained()` loads `model.safetensors`.
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
            model_config = RynnValueLangConfig.from_pretrained(
                config.model_id, revision=config.model_revision
            )
        else:
            model_config = RynnValueLangConfig.from_dict(config.model_config)
        model_config._attn_implementation = "pred_slot_isolated_eager"
        # Since transformers 5.22, Qwen3-VL passes the value-slot tensors on to the vision tower
        # too, which the official model runs without them, i.e. as plain eager attention.
        model_config.vision_config._attn_implementation = "eager"
        if config.model_config is None:
            self.model = RynnValueLangModel.from_pretrained(
                config.model_id, revision=config.model_revision, config=model_config, dtype=dtype
            )
        else:
            self.model = RynnValueLangModel._from_config(model_config, dtype=dtype)

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
        inputs: dict[str, Any] = {}
        for key in RYNNVALUE_MODEL_INPUT_KEYS:
            batch_key = f"{RYNNVALUE_FEATURE_PREFIX}{key}"
            if batch_key in batch:
                inputs[key] = batch[batch_key]
        if "input_ids" not in inputs:
            raise KeyError(
                f"RynnValue batch missing `{RYNNVALUE_FEATURE_PREFIX}input_ids`. "
                "Make sure RynnValueEncoderProcessorStep ran before predict_remaining_time()."
            )
        for required in ("attention_mask", "pixel_values", "image_grid_thw"):
            if required not in inputs:
                raise KeyError(f"RynnValue batch missing `{RYNNVALUE_FEATURE_PREFIX}{required}`.")

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
        return RemainingTimeOutput(remaining_time_s=remaining_time)

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config.model_config = self.model.config.to_dict()
        super()._save_pretrained(save_directory)
