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

from typing import Literal

from transformers import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


class ValueTokenizerConfig(PretrainedConfig):
    model_type = "value_tokenizer"

    def __init__(
        self,
        bins: int = 256,
        min_value: float = 0.0,
        max_value: float = 1000.0,
        support_transform: Literal["linear", "symlog", "quantile"] = "linear",
        encoding: Literal["two_hot", "hl_gauss"] = "two_hot",
        hl_gauss_sigma_ratio: float = 0.75,
        bin_edges: list[float] | None = None,
        bin_edges_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bins = bins
        self.min_value = min_value
        self.max_value = max_value
        self.support_transform = support_transform
        self.encoding = encoding
        self.hl_gauss_sigma_ratio = hl_gauss_sigma_ratio
        self.bin_edges_path = bin_edges_path
        self.bin_edges = list(bin_edges) if bin_edges is not None else None
        if support_transform == "quantile":
            if self.bin_edges is None:
                raise ValueError(
                    f"support_transform='quantile' requires explicit bin_edges (length bins+1={bins + 1})."
                )
            if len(self.bin_edges) != bins + 1:
                raise ValueError(
                    f"bin_edges must have length bins+1 ({bins + 1}), got {len(self.bin_edges)}."
                )
            if any(
                self.bin_edges[index + 1] <= self.bin_edges[index] for index in range(len(self.bin_edges) - 1)
            ):
                raise ValueError("bin_edges must be strictly increasing.")
        if encoding not in ("two_hot", "hl_gauss"):
            raise ValueError(f"Unsupported encoding: {encoding}. Expected 'two_hot' or 'hl_gauss'.")


class ValueHeadConfig(PretrainedConfig):
    model_type = "value_head"

    def __init__(
        self,
        head_type: str = "linear",
        hidden_dims: int = 1024,
        depth: int = 2,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_type = head_type
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.activation = activation


class RynnValueLangConfig(Qwen3VLConfig):
    model_type = "rynn_value_lang"
    sub_configs = {
        "vision_config": Qwen3VLVisionConfig,
        "text_config": Qwen3VLTextConfig,
        "value_tokenizer_config": ValueTokenizerConfig,
        "relative_value_tokenizer_config": ValueTokenizerConfig,
        "value_head_config": ValueHeadConfig,
        "relative_value_head_config": ValueHeadConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        value_tokenizer_config: ValueTokenizerConfig | dict | None = None,
        value_head_config: ValueHeadConfig | dict | None = None,
        relative_value_head_config: ValueHeadConfig | dict | None = None,
        relative_value_tokenizer_config: ValueTokenizerConfig | dict | None = None,
        num_value_heads: int = 1,
        value_token_repeat: int = 1,
        relative_value_token_repeat: int = 1,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._attn_implementation = attn_implementation or "pred_slot_isolated_eager"
        if value_tokenizer_config is None:
            value_tokenizer_config = ValueTokenizerConfig()
        elif isinstance(value_tokenizer_config, dict):
            value_tokenizer_config = ValueTokenizerConfig(**value_tokenizer_config)
        if relative_value_tokenizer_config is None and relative_value_head_config is not None:
            raise ValueError("relative_value_head_config is set but relative_value_tokenizer_config is None.")
        if isinstance(relative_value_tokenizer_config, dict):
            relative_value_tokenizer_config = ValueTokenizerConfig(**relative_value_tokenizer_config)
        if isinstance(value_head_config, dict):
            value_head_config = ValueHeadConfig(**value_head_config)
        if isinstance(relative_value_head_config, dict):
            relative_value_head_config = ValueHeadConfig(**relative_value_head_config)
        self.value_tokenizer_config = value_tokenizer_config
        self.relative_value_tokenizer_config = relative_value_tokenizer_config
        self.value_head_config = value_head_config
        self.relative_value_head_config = relative_value_head_config
        self.num_value_heads = int(num_value_heads)
        self.value_token_repeat = int(value_token_repeat)
        self.relative_value_token_repeat = int(relative_value_token_repeat)
        if self.value_token_repeat < 1:
            raise ValueError(f"value_token_repeat must be >= 1, got {self.value_token_repeat}")
        if self.relative_value_token_repeat < 1:
            raise ValueError(
                f"relative_value_token_repeat must be >= 1, got {self.relative_value_token_repeat}"
            )
        self.architectures = ["RynnValueLangModel"]

    @property
    def bins(self) -> int:
        return self.value_tokenizer_config.bins

    @classmethod
    def from_qwen3vl(cls, source: str | Qwen3VLConfig, **kwargs) -> "RynnValueLangConfig":
        if isinstance(source, str):
            base_config = Qwen3VLConfig.from_pretrained(source)
        elif isinstance(source, Qwen3VLConfig):
            base_config = source
        else:
            raise TypeError("source must be a pretrained model name/path or a Qwen3VLConfig instance.")
        base_dict = base_config.to_dict()
        for key in (
            "model_type",
            "value_tokenizer_config",
            "relative_value_tokenizer_config",
            "value_head_config",
            "relative_value_head_config",
            "success_head_config",
            "match_head_config",
            "architectures",
            "bins",
            "num_value_heads",
            "value_token_repeat",
            "relative_value_token_repeat",
        ):
            base_dict.pop(key, None)
        base_dict.update(kwargs)
        return cls(**base_dict)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        persisted_attn = config_dict.get("attn_implementation")
        outputs = super().from_dict(config_dict, **kwargs)
        config = outputs[0] if isinstance(outputs, tuple) else outputs
        if persisted_attn is not None and kwargs.get("attn_implementation") is None:
            config._attn_implementation = persisted_attn
        return outputs

    def to_dict(self):
        output = super().to_dict()
        output.update(
            architectures=["RynnValueLangModel"],
            value_tokenizer_config=self.value_tokenizer_config.to_dict(),
            relative_value_tokenizer_config=(
                self.relative_value_tokenizer_config.to_dict()
                if self.relative_value_tokenizer_config is not None
                else None
            ),
            value_head_config=(
                self.value_head_config.to_dict() if self.value_head_config is not None else None
            ),
            relative_value_head_config=(
                self.relative_value_head_config.to_dict()
                if self.relative_value_head_config is not None
                else None
            ),
            num_value_heads=self.num_value_heads,
            value_token_repeat=self.value_token_repeat,
            relative_value_token_repeat=self.relative_value_token_repeat,
        )
        if getattr(self, "_attn_implementation", None) is not None:
            output["attn_implementation"] = self._attn_implementation
        return output
