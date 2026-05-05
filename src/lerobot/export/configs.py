#!/usr/bin/env python

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
"""Export configuration dataclasses authored by policies and consumed by runners.

Each supported runner family has a matching config dataclass:

- :class:`SinglePassExportConfig`    -> :class:`~lerobot.export.runners.single_pass.SinglePassRunner`
- :class:`KVCacheFlowExportConfig`   -> :class:`~lerobot.export.runners.kv_cache_flow.KVCacheFlowRunner`
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SinglePassExportConfig:
    """Configuration for single-pass (single forward pass) export.

    Used by policies like ACT and VQ-BeT that produce actions in one forward pass.
    """

    chunk_size: int
    action_dim: int
    n_action_steps: int | None = None


@dataclass
class KVCacheFlowExportConfig:
    """Configuration for KV-cache flow-matching (VLA) export.

    Captures architecture-specific information needed to export a KV-cache
    flow-matching policy (e.g. PI05) and reconstruct the KV cache at runtime.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int

    chunk_size: int
    action_dim: int
    state_dim: int | None
    num_steps: int

    input_mapping: dict[str, str] = field(default_factory=dict)


ExportConfig = SinglePassExportConfig | KVCacheFlowExportConfig


__all__ = [
    "ExportConfig",
    "KVCacheFlowExportConfig",
    "SinglePassExportConfig",
]
