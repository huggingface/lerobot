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

"""Safe serialization replacing pickle with safetensors + JSON.

The async inference pipeline previously used `pickle` for serializing data
over gRPC, which allows arbitrary code execution when deserializing
untrusted data (CWE-502). This module replaces all pickle usage with:

- JSON for scalar metadata and configuration
- safetensors for tensor data

This eliminates the risk of remote code execution through crafted payloads
while maintaining full compatibility with the existing data flow.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from lerobot.configs.types import FeatureType, PolicyFeature

from .helpers import RemotePolicyConfig, TimedAction, TimedObservation


def _pack(metadata: dict[str, Any], tensors: dict[str, torch.Tensor] | None = None) -> bytes:
    """Pack JSON metadata and optional safetensors data into a byte stream.

    Wire format: [4-byte JSON length (big-endian uint32)] [JSON bytes] [safetensors bytes]
    """
    json_bytes = json.dumps(metadata).encode("utf-8")
    header = struct.pack(">I", len(json_bytes))
    tensor_bytes = st_save(tensors) if tensors else b""
    return header + json_bytes + tensor_bytes


def _unpack(data: bytes) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Unpack a byte stream into JSON metadata and a tensor dict."""
    json_length = struct.unpack(">I", data[:4])[0]
    json_data = json.loads(data[4 : 4 + json_length].decode("utf-8"))
    tensor_data = data[4 + json_length :]
    tensors = st_load(tensor_data) if tensor_data else {}
    return json_data, tensors


# ---------------------------------------------------------------------------
# RemotePolicyConfig
# ---------------------------------------------------------------------------


def serialize_policy_config(config: RemotePolicyConfig) -> bytes:
    """Serialize a RemotePolicyConfig to bytes (JSON only, no tensors)."""
    metadata = {
        "type": "RemotePolicyConfig",
        "policy_type": config.policy_type,
        "pretrained_name_or_path": config.pretrained_name_or_path,
        "lerobot_features": {
            k: (
                {"type": v.type.value, "shape": list(v.shape)}
                if isinstance(v, PolicyFeature)
                else v  # already a plain dict (e.g. from hw_to_dataset_features)
            )
            for k, v in config.lerobot_features.items()
        },
        "actions_per_chunk": config.actions_per_chunk,
        "device": config.device,
        "rename_map": config.rename_map,
    }
    return _pack(metadata)


def deserialize_policy_config(data: bytes) -> RemotePolicyConfig:
    """Deserialize bytes into a RemotePolicyConfig."""
    meta, _ = _unpack(data)
    if meta.get("type") != "RemotePolicyConfig":
        raise ValueError(f"Expected RemotePolicyConfig, got {meta.get('type')}")
    return RemotePolicyConfig(
        policy_type=meta["policy_type"],
        pretrained_name_or_path=meta["pretrained_name_or_path"],
        lerobot_features={
            k: (
                PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                if "type" in v and v["type"] in {e.value for e in FeatureType}
                else v  # pass through plain dicts as-is
            )
            for k, v in meta["lerobot_features"].items()
        },
        actions_per_chunk=meta["actions_per_chunk"],
        device=meta.get("device", "cpu"),
        rename_map=meta.get("rename_map", {}),
    )


# ---------------------------------------------------------------------------
# TimedObservation
# ---------------------------------------------------------------------------


def serialize_observation(obs: TimedObservation) -> bytes:
    """Serialize a TimedObservation using safetensors for tensors and JSON for scalars."""
    tensors: dict[str, torch.Tensor] = {}
    scalar_data: dict[str, Any] = {}
    numpy_keys: list[str] = []

    for key, value in obs.get_observation().items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.contiguous()
        elif isinstance(value, np.ndarray):
            tensors[key] = torch.from_numpy(np.ascontiguousarray(value))
            numpy_keys.append(key)
        else:
            # str, int, float, list, dict, bool, None - all JSON-safe
            scalar_data[key] = value

    metadata = {
        "type": "TimedObservation",
        "timestamp": obs.get_timestamp(),
        "timestep": obs.get_timestep(),
        "must_go": obs.must_go,
        "scalar_data": scalar_data,
        "numpy_keys": numpy_keys,
    }
    return _pack(metadata, tensors if tensors else None)


def deserialize_observation(data: bytes) -> TimedObservation:
    """Deserialize bytes into a TimedObservation."""
    meta, tensors = _unpack(data)
    if meta.get("type") != "TimedObservation":
        raise ValueError(f"Expected TimedObservation, got {meta.get('type')}")

    numpy_keys = set(meta.get("numpy_keys", []))
    observation: dict[str, Any] = dict(meta["scalar_data"])

    for key, tensor in tensors.items():
        if key in numpy_keys:
            observation[key] = tensor.numpy()
        else:
            observation[key] = tensor

    return TimedObservation(
        timestamp=meta["timestamp"],
        timestep=meta["timestep"],
        observation=observation,
        must_go=meta.get("must_go", False),
    )


# ---------------------------------------------------------------------------
# list[TimedAction]
# ---------------------------------------------------------------------------


def serialize_actions(actions: list[TimedAction]) -> bytes:
    """Serialize a list of TimedAction using safetensors for action tensors."""
    tensors: dict[str, torch.Tensor] = {}
    action_metadata: list[dict[str, float | int]] = []

    for i, action in enumerate(actions):
        tensor = action.get_action()
        tensors[f"action_{i}"] = tensor.detach().contiguous()
        action_metadata.append(
            {
                "timestamp": action.get_timestamp(),
                "timestep": action.get_timestep(),
            }
        )

    metadata = {
        "type": "TimedActions",
        "actions": action_metadata,
    }
    return _pack(metadata, tensors if tensors else None)


def deserialize_actions(data: bytes) -> list[TimedAction]:
    """Deserialize bytes into a list of TimedAction."""
    meta, tensors = _unpack(data)
    if meta.get("type") != "TimedActions":
        raise ValueError(f"Expected TimedActions, got {meta.get('type')}")

    actions: list[TimedAction] = []
    for i, action_meta in enumerate(meta["actions"]):
        actions.append(
            TimedAction(
                timestamp=action_meta["timestamp"],
                timestep=action_meta["timestep"],
                action=tensors[f"action_{i}"],
            )
        )
    return actions
