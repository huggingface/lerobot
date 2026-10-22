#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""Wall-X configuration extensions for the native Transformers Qwen2.5-VL config."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from huggingface_hub.dataclasses import strict

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig as TransformersQwen2_5_VLConfig,
        Qwen2_5_VLTextConfig as TransformersQwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
else:

    @dataclass
    class _TransformersConfigFallback:
        """Import-safe stand-in used only when Transformers is unavailable."""

    TransformersQwen2_5_VLConfig = _TransformersConfigFallback
    TransformersQwen2_5_VLTextConfig = _TransformersConfigFallback
    Qwen2_5_VLVisionConfig = None

# Wall-X checkpoints pre0.6.0 use the legacy, flat Qwen2.5-VL config layout.  The native
# ``Qwen2_5_VLConfig`` accepts that layout and moves text-model fields into its
# ``text_config`` sub-config, so only the Wall-X-specific MoE fields need to be
# declared here.
_LEGACY_TEXT_ATTRIBUTES = {
    "attention_dropout",
    "attention_moe",
    "dim_inputs",
    "dof_config",
    "experts",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "max_window_layers",
    "mlp_moe",
    "noise_scheduler",
    "num_attention_heads",
    "num_experts",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "sliding_window",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
}


@strict
class Qwen2_5_VLTextConfig(TransformersQwen2_5_VLTextConfig):  # noqa: N801
    """Native Qwen2.5-VL text config plus Wall-X's hard-routed MoE settings."""

    num_experts: int = 4
    experts: list[dict] | None = None
    dof_config: dict | None = None
    noise_scheduler: dict | None = None
    dim_inputs: tuple[int, ...] | list[int] = (1536, 1536)
    attention_moe: bool = False
    mlp_moe: bool = False

    def __post_init__(self, **kwargs):
        self.dim_inputs = tuple(self.dim_inputs)
        super().__post_init__(**kwargs)


@strict
class Qwen2_5_VLConfig(TransformersQwen2_5_VLConfig):  # noqa: N801
    """Native composite Qwen2.5-VL config with a Wall-X text sub-config.

    The native composite loader supports both current nested configs and the
    flat layout used by existing ``wall-oss-flow`` checkpoints.
    """

    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        "text_config": Qwen2_5_VLTextConfig,
    }

    def __getattr__(self, name):
        """Keep legacy direct access to fields now owned by ``text_config``.

        Wall-X historically used a flat config and accesses fields such as
        ``hidden_size`` and ``num_experts`` directly. Forwarding unknown
        attributes preserves that API without duplicating the native config.
        """
        text_config = self.__dict__.get("text_config")
        if name in _LEGACY_TEXT_ATTRIBUTES and text_config is not None and hasattr(text_config, name):
            return getattr(text_config, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
