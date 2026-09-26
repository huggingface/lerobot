#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Sequence
from typing import Any

import torch

from lerobot.utils.import_utils import require_package


def flatten_feature_names(names: Any) -> list[str] | None:
    """Flatten LeRobot's flat or grouped feature-name metadata."""
    if names is None:
        return None
    if isinstance(names, str):
        return [names]
    if isinstance(names, dict):
        indexed_names = list(names.items())
        if indexed_names and all(
            isinstance(index, int) and not isinstance(index, bool) for _, index in indexed_names
        ):
            indexed_names.sort(key=lambda item: item[1])
            if [index for _, index in indexed_names] != list(range(len(indexed_names))):
                return None
            return [str(name) for name, _ in indexed_names]
        values = names.values()
    elif isinstance(names, Sequence):
        values = names
    else:
        return None

    flattened = []
    for value in values:
        nested = flatten_feature_names(value)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened or None


def import_dm05_core():
    """Import the self-contained DM05 core bundled with this LeRobot policy."""
    require_package("transformers", extra="dm05")
    from .modeling import DM05CoreModelConfig, DM05ForCausalLM

    return DM05CoreModelConfig, DM05ForCausalLM


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    """Resolve a DM05 dtype string to a torch dtype."""
    if dtype in {"bfloat16", "float32"}:
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def normalize_task_batch(task: Any, batch_size: int, default_task: str) -> list[str]:
    """Broadcast or validate task prompts for a batched DM05 input."""
    if task is None:
        return [default_task] * batch_size
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, Sequence) and len(values := list(task)) in {1, batch_size}:
        return [str(values[0 if len(values) == 1 else idx]) for idx in range(batch_size)]
    raise ValueError(f"Cannot broadcast task={task!r} to batch_size={batch_size}")


def build_meta(image_keys: Sequence[str]) -> dict[str, Any]:
    """Build the minimal metadata consumed by the DM05 prompt renderer."""
    return {"dataset_meta": {"image_keys": list(image_keys)}}
