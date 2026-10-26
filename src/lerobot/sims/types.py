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

"""Transport-free simulation values and environment metadata."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimConfig:
    """The simulator an environment runs: its integration, task, options and control conventions."""

    type: str = "toy"
    task: str = "reach"
    fps: float = 20
    control: str = "delta"
    semantics: str = "toy-delta-v1"
    kwargs: dict[str, Any] = field(default_factory=dict)
    state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    gripper_indices: tuple[int, ...] = ()
    command_retention_indices: tuple[int, ...] = ()
    robot_type: str = ""

    def __post_init__(self) -> None:
        """Normalize component order and reject invalid controller declarations."""
        self.state_names = tuple(self.state_names)
        self.action_names = tuple(self.action_names)
        self.gripper_indices = tuple(self.gripper_indices)
        self.command_retention_indices = tuple(self.command_retention_indices)
        if (
            self.control not in {"position", "delta", "eef_pose", "velocity"}
            or not self.semantics
            or not np.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("A simulator requires explicit semantics, controller, and positive FPS")


@dataclass(frozen=True)
class FeatureSpec:
    """Shape, dtype, components, and semantics of one environment value."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "tensor"
    names: tuple[str, ...] = ()
    semantics: str = ""

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported feature metadata."""
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "names", tuple(self.names))
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.semantics, str)
            or not self.semantics.strip()
        ):
            raise ValueError("Feature names and explicit semantic conventions are required.")
        if self.kind not in {"tensor", "rgb"}:
            raise ValueError(f"Unsupported feature modality: {self.kind!r}")
        if len(self.shape) > 4 or any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError(f"Invalid feature shape: {self.shape!r}")
        if self.dtype not in {"bool", "uint8", "int32", "int64", "float32", "float64"}:
            raise ValueError(f"Unsupported feature dtype: {self.dtype!r}")
        if self.kind == "rgb" and (len(self.shape) != 3 or self.shape[-1] != 3 or self.dtype != "uint8"):
            raise ValueError("RGB features require HWC uint8 with three channels.")
        if self.names and (len(self.shape) != 1 or len(self.names) != self.shape[0]):
            raise ValueError("Ordered component names must match a one-dimensional feature.")
        if any(not isinstance(name, str) or not name.strip() for name in self.names):
            raise ValueError("Feature component names must be nonempty strings.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature component names must be unique.")


@dataclass(frozen=True)
class EnvDescriptor:
    """Canonical spaces, control convention, and tasks of an environment."""

    sim_type: str
    semantics: str
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    control: str
    fps: float
    tasks: dict[str, list[str]]
    max_episode_steps: int
    sim_build: dict[str, str]
    gripper_indices: tuple[int, ...] = ()
    command_retention_indices: tuple[int, ...] = ()
    robot_type: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid environment metadata."""
        if self.control not in {"position", "delta", "eef_pose", "velocity"}:
            raise ValueError("Unknown controller convention")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("FPS and episode limit must be positive")
        if len(self.action_feature.shape) != 1 or not self.action_feature.names:
            raise ValueError("Actions require ordered component names")
        if len({feature.name for feature in self.features}) != len(self.features):
            raise ValueError("Duplicate observation features")
        if any(
            index < 0 or index >= self.action_feature.shape[0]
            for index in self.gripper_indices + self.command_retention_indices
        ):
            raise ValueError("Invalid gripper component")


@dataclass(frozen=True)
class StepResult:
    """Result of stepping a batch of environments once."""

    obs: dict[str, np.ndarray]
    task: tuple[str, ...]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    is_success: np.ndarray
    sim_time: float
    step: np.ndarray
