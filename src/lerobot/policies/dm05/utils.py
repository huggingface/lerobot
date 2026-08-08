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
import torch.nn.functional as torch_nn_functional
from PIL import Image
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

JOINT_ID = 0
GRIPPER_ID = 2


def import_dm05_core():
    """Import the self-contained DM05 core bundled with this LeRobot policy."""
    require_package("transformers", extra="dm05")
    from .modeling_dm05_core import DM05CoreModelConfig, DM05ForCausalLM
    from .tokenization_dm05 import DM05Tokenization

    return DM05CoreModelConfig, DM05ForCausalLM, DM05Tokenization


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype in {"bfloat16", "float32"}:
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def pad_vector(x: Tensor, dim: int) -> Tensor:
    return x[..., :dim] if x.shape[-1] >= dim else torch_nn_functional.pad(x, (0, dim - x.shape[-1]))


def pad_action_chunk(action: Tensor, chunk_size: int, action_dim: int) -> Tensor:
    if action.dim() == 1:
        action = action.unsqueeze(0)
    if action.shape[0] > chunk_size:
        action = action[:chunk_size]
    action = pad_vector(action, action_dim)
    if action.shape[0] < chunk_size:
        action = torch_nn_functional.pad(action, (0, 0, 0, chunk_size - action.shape[0]))
    return action


def tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().cpu()
    if image.dim() != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(image.shape)}")
    if image.shape[0] in {1, 3, 4}:
        image = image.permute(1, 2, 0)
    if image.dtype.is_floating_point:
        image = image.clamp(0, 1) * 255
    arr = image.to(torch.uint8).numpy()
    return Image.fromarray(arr[..., 0] if arr.ndim == 3 and arr.shape[-1] == 1 else arr).convert("RGB")


def normalize_task_batch(task: Any, batch_size: int, default_task: str) -> list[str]:
    if task is None:
        return [default_task] * batch_size
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, Sequence) and len(values := list(task)) in {1, batch_size}:
        return [str(values[0 if len(values) == 1 else idx]) for idx in range(batch_size)]
    raise ValueError(f"Cannot broadcast task={task!r} to batch_size={batch_size}")


def get_image_keys(batch: dict[str, Any], configured_keys: Sequence[str] | None = None) -> list[str]:
    if configured_keys:
        return [key for key in configured_keys if key in batch]
    keys = [key for key in batch if key.startswith(f"{OBS_IMAGES}.")]
    return sorted(keys or (["observation.image"] if "observation.image" in batch else []))


def infer_dm05_state_desc(config: Any) -> tuple[int, ...]:
    features = getattr(config, "input_features", None)
    if not (shape := getattr(features.get(OBS_STATE) if isinstance(features, dict) else None, "shape", None)):
        return ()

    if (gripper_indices := {7: (6,), 8: (6, 7), 14: (6, 13)}.get(int(shape[-1]))) is None:
        return ()

    return tuple(GRIPPER_ID if idx in gripper_indices else JOINT_ID for idx in range(int(shape[-1])))


def build_meta(config: Any, image_keys: Sequence[str]) -> dict[str, Any]:
    meta = {"dataset_meta": {"image_keys": list(image_keys)}}
    if state_desc := infer_dm05_state_desc(config):
        meta["state_desc"] = list(state_desc)
        meta["dataset_meta"]["state_desc"] = list(state_desc)
    if non_delta_indices := infer_dm05_non_delta_indices(config, meta):
        meta["non_delta_indices"] = list(non_delta_indices)
    return meta


def dm05_non_delta_indices_from_desc(state_desc: Sequence[Any]) -> tuple[int, ...]:
    return tuple(
        idx
        for idx, desc in enumerate(state_desc)
        for value in (getattr(desc, "value", desc),)
        if (value.lower() == "gripper" if isinstance(value, str) else int(value) == GRIPPER_ID)
    )


def infer_dm05_non_delta_indices(config: Any, meta_data: dict[str, Any] | None = None) -> tuple[int, ...]:
    if (explicit := getattr(config, "norm_non_delta_indices", None)) is not None:
        return tuple(int(idx) for idx in explicit)
    if isinstance(meta_data, dict):
        if (meta_indices := meta_data.get("non_delta_indices")) is not None:
            return tuple(int(idx) for idx in meta_indices)
        state_desc = meta_data.get("state_desc")
        if state_desc is None:
            dataset_meta = meta_data.get("dataset_meta")
            if isinstance(dataset_meta, dict):
                state_desc = dataset_meta.get("state_desc")
        if state_desc is not None:
            return dm05_non_delta_indices_from_desc(state_desc)
    return dm05_non_delta_indices_from_desc(infer_dm05_state_desc(config))


def collate_dm05_instances(
    instances: Sequence[dict[str, Tensor]],
    *,
    pad_token_id: int,
    max_length: int | None,
) -> dict[str, Tensor]:
    if not instances:
        raise ValueError("Cannot collate an empty DM05 batch")

    seq_lens = [int(inst["input_ids"].shape[0]) for inst in instances]
    target_len = max(seq_lens) if max_length is None else min(max(seq_lens), int(max_length))

    batch: dict[str, Tensor] = {}
    for key, pad_value, dtype in (
        ("input_ids", pad_token_id, torch.long),
        ("attention_mask", 0, torch.long),
        ("labels", -100, torch.long),
        ("token_type_ids", 0, torch.long),
    ):
        if not any(key in inst for inst in instances):
            continue
        values = []
        for inst in instances:
            value = inst[key][:target_len] if key in inst else torch.full((0,), pad_value, dtype=dtype)
            pad_len = target_len - value.shape[0]
            if pad_len > 0:
                value = torch.cat(
                    [value, torch.full((pad_len,), pad_value, dtype=value.dtype, device=value.device)]
                )
            values.append(value)
        batch[key] = torch.stack(values, dim=0)

    if any("pixel_values" in inst for inst in instances):
        batch["pixel_values"] = torch.cat(
            [inst["pixel_values"] for inst in instances if "pixel_values" in inst],
            dim=0,
        )

    for key in ("states", "actions", "action_dim_mask", "has_actions"):
        if any(key in inst for inst in instances):
            batch[key] = torch.stack([inst[key] for inst in instances], dim=0)

    return batch
