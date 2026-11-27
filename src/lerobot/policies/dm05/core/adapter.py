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

from lerobot.utils.import_utils import require_package


def import_dm05_core():
    """Import the self-contained DM05 core bundled with this LeRobot policy."""
    require_package("transformers", extra="dm05")
    from .modeling import DM05CoreModelConfig, DM05ForCausalLM

    return DM05CoreModelConfig, DM05ForCausalLM


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
