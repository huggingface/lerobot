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

from lerobot.utils.constants import OBS_IMAGES
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
    from .modeling_dm05_core import DM05CoreModelConfig, DM05ForCausalLM

    return DM05CoreModelConfig, DM05ForCausalLM


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    """Resolve a DM05 dtype string to a torch dtype."""
    if dtype in {"bfloat16", "float32"}:
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def relative_action_mask(
    action_dim: int,
    action_names: Sequence[str] | None,
    exclude_joints: Sequence[str],
) -> list[bool]:
    """Return the OpenDM-style delta mask for an action vector."""
    excluded = [str(name).lower() for name in exclude_joints if name]
    if not excluded:
        return [True] * action_dim
    if action_names is None or len(action_names) != action_dim:
        raise ValueError(
            "DM05 relative_exclude_joints requires one action feature name per dimension; "
            f"got {0 if action_names is None else len(action_names)} names for {action_dim} dimensions."
        )

    action_names_lower = [str(name).lower() for name in action_names]
    unmatched = [
        token for token in excluded if not any(token == name or token in name for name in action_names_lower)
    ]
    if unmatched:
        raise ValueError(f"DM05 relative_exclude_joints did not match action feature names: {unmatched}.")
    mask = [not any(token == name or token in name for token in excluded) for name in action_names_lower]
    return mask


def build_action_prefix_mask(
    action_prefill_len: torch.Tensor | None,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a mask for timesteps that are fixed by action prefill."""
    if action_prefill_len is None:
        return None

    lengths = action_prefill_len.to(device=device, dtype=torch.long)
    positions = torch.arange(horizon, device=device)
    return positions[None, :] < lengths[:, None]


def validate_action_prefill_pair(
    prefill_actions: torch.Tensor | None,
    action_prefill_len: torch.Tensor | None,
) -> None:
    """Require action prefill values and lengths to be provided together."""
    if (prefill_actions is None) != (action_prefill_len is None):
        raise ValueError("prefill_actions and action_prefill_len must be provided together.")


def normalize_task_batch(task: Any, batch_size: int, default_task: str) -> list[str]:
    """Broadcast or validate task prompts for a batched DM05 input."""
    if task is None:
        return [default_task] * batch_size
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, Sequence) and len(values := list(task)) in {1, batch_size}:
        return [str(values[0 if len(values) == 1 else idx]) for idx in range(batch_size)]
    raise ValueError(f"Cannot broadcast task={task!r} to batch_size={batch_size}")


def get_image_keys(batch: dict[str, Any], configured_keys: Sequence[str] | None = None) -> list[str]:
    """Resolve the ordered image observation keys used by DM05."""
    if configured_keys:
        return [key for key in configured_keys if key in batch]
    keys = [key for key in batch if key.startswith(f"{OBS_IMAGES}.")]
    return sorted(keys or (["observation.image"] if "observation.image" in batch else []))


def build_meta(image_keys: Sequence[str]) -> dict[str, Any]:
    """Build the minimal metadata consumed by the DM05 prompt renderer."""
    return {"dataset_meta": {"image_keys": list(image_keys)}}
