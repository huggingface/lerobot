#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_FILENAME = "rl_token_config.json"


@dataclass(frozen=True)
class RLTokenConfig:
    """Configuration for the Stage 1 RL-token information bottleneck."""

    vla_dim: int
    token_dim: int = 256
    max_tokens: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vla_dim <= 0 or self.token_dim <= 0:
            raise ValueError("vla_dim and token_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.token_dim % self.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if self.mlp_ratio <= 0.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("mlp_ratio must be positive and dropout must be in [0, 1)")

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / CONFIG_FILENAME, "w", encoding="utf-8") as config_file:
            json.dump(asdict(self), config_file, indent=2)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> RLTokenConfig:
        with open(Path(directory) / CONFIG_FILENAME, encoding="utf-8") as config_file:
            return cls(**json.load(config_file))
