# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Dict, List
from collections.abc import Mapping

import torch


class AttentionRule(Enum):
    """Enum describing when to attend to another token group."""

    NEVER = "never"
    CAUSAL = "other.timestep <= self.timestep"
    CURRENT = "other.timestep == self.timestep"
    STRICT_PAST = "other.timestep < self.timestep"
    ALL = "all"


RULE_MAP = {
    AttentionRule.NEVER: 0,
    AttentionRule.CAUSAL: 1,
    AttentionRule.CURRENT: 2,
    AttentionRule.STRICT_PAST: 3,
    AttentionRule.ALL: 4,
}


def find_match(pattern_dict: dict[str, Any], name: str, default: Any) -> Any:
    """Find the first matching pattern in the dictionary, or return the default value."""
    for pattern, value in pattern_dict.items():
        if fnmatch(name, pattern):
            return value
    return default


@dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)."""

    tokens: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.mask.ndim != self.tokens.ndim - 1:
            raise ValueError(
                f"Mask must have one less dimension than tokens, "
                f"but got {self.mask.ndim} and {self.tokens.ndim}"
            )

    @classmethod
    def concatenate(cls, group_list: list["TokenGroup"], axis: int = -2) -> "TokenGroup":
        """Concatenates a list of TokenGroups along a specified axis."""
        if not group_list:
            raise ValueError("Cannot concatenate an empty list of TokenGroups")

        tokens = torch.cat([t.tokens for t in group_list], dim=axis)
        mask = torch.cat([t.mask for t in group_list], dim=axis + 1)

        return cls(tokens=tokens, mask=mask)


@dataclass
class PrefixGroup(TokenGroup):
    """A group of tokens that will be at the beginning of the token sequence (e.g. task tokens)."""

    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        super().__post_init__()
        if len(self.tokens.shape) != 3:
            raise ValueError(
                f"PrefixGroup tokens must be (batch, n_tokens, d), but got shape {self.tokens.shape}"
            )
        if len(self.mask.shape) != 2:
            raise ValueError(f"PrefixGroup mask must be (batch, n_tokens), but got shape {self.mask.shape}")


@dataclass
class TimestepGroup(TokenGroup):
    """A group of tokens that is repeated for each timestep (e.g. observation tokens)."""

    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        super().__post_init__()
        if len(self.tokens.shape) != 4:
            raise ValueError(
                f"TimestepGroup tokens must be (batch, horizon, n_tokens, d), "
                f"but got shape {self.tokens.shape}"
            )
        if len(self.mask.shape) != 3:
            raise ValueError(
                f"TimestepGroup mask must be (batch, horizon, n_tokens), but got shape {self.mask.shape}"
            )
