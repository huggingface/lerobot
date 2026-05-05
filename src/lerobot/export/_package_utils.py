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

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .manifest import CameraConfig, HardwareConfig, RobotConfig, TensorSpec

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "build_hardware_config",
    "generate_example_batch",
    "get_action_dim",
    "get_normalization_groups",
    "get_normalized_input_features",
    "get_policy_stats",
    "is_json_serializable",
    "save_policy_config",
]


def generate_example_batch(policy: PreTrainedPolicy) -> dict[str, Tensor]:
    """Build a random single-item batch matching the policy's expected input features.

    Constructs tensors of the correct shape for each declared feature (robot state,
    environment state, and camera images) on the same device as the policy parameters.

    Args:
        policy: Instantiated policy whose ``config`` declares the input features.

    Returns:
        Dict mapping feature keys (e.g. ``"observation.state"``) to random float tensors
        with batch size 1.
    """
    config = policy.config
    batch_size = 1
    device = next(policy.parameters()).device
    batch: dict[str, Tensor] = {}
    if hasattr(config, "robot_state_feature") and config.robot_state_feature:
        batch["observation.state"] = torch.randn(
            batch_size, config.robot_state_feature.shape[0], device=device
        )
    if hasattr(config, "env_state_feature") and config.env_state_feature:
        batch["observation.environment_state"] = torch.randn(
            batch_size, config.env_state_feature.shape[0], device=device
        )
    if hasattr(config, "image_features") and config.image_features:
        for img_key, img_feature in config.image_features.items():
            batch[img_key] = torch.randn(batch_size, *img_feature.shape, device=device)
    return batch


def build_hardware_config(policy: PreTrainedPolicy) -> HardwareConfig | None:
    """Derive a :class:`HardwareConfig` from the policy's declared features.

    Inspects ``policy.config`` for robot state, environment state, and image features
    and maps them to :class:`RobotConfig` / :class:`CameraConfig` entries.

    Args:
        policy: Instantiated policy with a ``config`` attribute.

    Returns:
        A populated :class:`HardwareConfig`, or ``None`` if no hardware features are declared.
    """
    config = policy.config
    robots: list[RobotConfig] = []
    cameras: list[CameraConfig] = []
    action_dim = get_action_dim(config)
    state_dim = (
        config.robot_state_feature.shape[0]
        if hasattr(config, "robot_state_feature") and config.robot_state_feature
        else None
    )
    if state_dim or action_dim:
        robot = RobotConfig(name="main")
        if state_dim:
            robot.state = TensorSpec(shape=[state_dim], dtype="float32")
        if action_dim:
            robot.action = TensorSpec(shape=[action_dim], dtype="float32")
        robots.append(robot)
    if hasattr(config, "image_features") and config.image_features:
        for img_key, img_feature in config.image_features.items():
            cameras.append(
                CameraConfig(name=img_key.split(".")[-1], shape=list(img_feature.shape), dtype="uint8")
            )
    if not robots and not cameras:
        return None
    return HardwareConfig(robots=robots or None, cameras=cameras or None)


def get_action_dim(config: Any) -> int:
    """Return the action dimension declared in a policy config.

    Checks ``max_action_dim`` first (used by variable-action policies), then falls back
    to ``action_feature.shape[0]``.

    Args:
        config: Policy config object.

    Returns:
        Integer action dimension.

    Raises:
        ValueError: If neither ``max_action_dim`` nor ``action_feature`` is set.
    """
    if hasattr(config, "max_action_dim") and config.max_action_dim is not None:
        return config.max_action_dim
    if hasattr(config, "action_feature") and config.action_feature is not None:
        return config.action_feature.shape[0]
    raise ValueError(
        f"Cannot determine action dimension for config of type {type(config).__name__}: "
        "neither `max_action_dim` nor `action_feature` is set."
    )


def get_policy_stats(policy: PreTrainedPolicy) -> dict[str, dict[str, Any]] | None:
    """Extract normalisation statistics from a policy, if available.

    Searches ``policy.policy_processor.steps`` for a step with a ``stats`` attribute,
    then falls back to ``policy.config.stats``.

    Args:
        policy: Instantiated policy to inspect.

    Returns:
        Nested stats dict ``{feature_key: {stat_name: value}}``, or ``None`` if not found.
    """
    if hasattr(policy, "policy_processor"):
        for step in getattr(policy.policy_processor, "steps", []):
            if hasattr(step, "stats") and step.stats:
                return step.stats
    if hasattr(policy, "config") and hasattr(policy.config, "stats"):
        return policy.config.stats
    return None


def _feature_key_to_type(policy: PreTrainedPolicy, feature_key: str) -> str | None:
    cfg = policy.config
    if feature_key == "observation.state" and getattr(cfg, "robot_state_feature", None):
        return "STATE"
    if feature_key == "observation.environment_state" and getattr(cfg, "env_state_feature", None):
        return "ENV"
    image_features = getattr(cfg, "image_features", None) or {}
    if feature_key in image_features:
        return "VISUAL"
    if feature_key == "action":
        return "ACTION"
    return None


def get_normalization_groups(
    policy: PreTrainedPolicy, feature_keys: list[str]
) -> list[tuple[str, list[str]]]:
    """Group feature keys by their normalization mode as declared in ``config.normalization_mapping``.

    Returns a list of ``(mode_str, [feature_keys])`` tuples. Features without a mapping or with
    ``IDENTITY`` are dropped (no normalization needed).
    """
    mapping = getattr(policy.config, "normalization_mapping", {}) or {}
    groups: dict[str, list[str]] = {}
    for key in feature_keys:
        ftype = _feature_key_to_type(policy, key)
        if ftype is None:
            continue
        mode = mapping.get(ftype)
        if mode is None:
            continue
        mode_value = mode.value if hasattr(mode, "value") else str(mode)
        mode_key = mode_value.lower()
        if mode_key == "identity":
            continue
        groups.setdefault(mode_key, []).append(key)
    return list(groups.items())


def get_normalized_input_features(policy: PreTrainedPolicy) -> list[str]:
    """Return the ordered list of input feature keys that require normalisation.

    Includes robot state, environment state, and all image feature keys declared in
    ``policy.config``, in that order.

    Args:
        policy: Instantiated policy with a ``config`` attribute.

    Returns:
        List of feature key strings (e.g. ``["observation.state", "observation.images.top"]``).
    """
    cfg = policy.config
    out: list[str] = []
    if getattr(cfg, "robot_state_feature", None):
        out.append("observation.state")
    if getattr(cfg, "env_state_feature", None):
        out.append("observation.environment_state")
    image_features = getattr(cfg, "image_features", None) or {}
    out.extend(image_features.keys())
    return out


def save_policy_config(policy: PreTrainedPolicy, path: Path) -> None:
    """Serialise JSON-serialisable fields of ``policy.config`` to a file.

    Non-serialisable fields are silently dropped. Errors during write are logged as
    warnings rather than raised, so export can continue without a config snapshot.

    Args:
        policy: Policy whose ``config.__dict__`` is serialised.
        path: Destination file path (created or overwritten).
    """
    try:
        config_dict = {k: v for k, v in policy.config.__dict__.copy().items() if is_json_serializable(v)}
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Could not save policy config to %s: %s", path, e)


def is_json_serializable(value: Any) -> bool:
    """Return ``True`` if ``value`` can be round-tripped through ``json.dumps``.

    Args:
        value: Any Python object to test.

    Returns:
        ``True`` if serialisable, ``False`` otherwise.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
