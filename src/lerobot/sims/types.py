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

from lerobot.utils.feature_spec import FeatureSpec

_EMPTY_CAMERA_PREFIX = "observation.images.empty_camera_"


@dataclass(frozen=True)
class EnvDescriptor:
    """Canonical spaces, control convention, and tasks of an environment."""

    sim_type: str
    semantics: str
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    control_mode: str
    fps: float
    tasks: dict[str, list[str]]
    max_episode_steps: int
    robot_type: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid environment metadata."""
        if self.control_mode not in {"position", "delta", "eef_pose"}:
            raise ValueError("Unknown controller convention")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("FPS and episode limit must be positive")
        if len(self.action_feature.shape) != 1 or not self.action_feature.names:
            raise ValueError("Actions require ordered component names")
        if len({feature.name for feature in self.features}) != len(self.features):
            raise ValueError("Duplicate observation features")

    def validate_checkpoint(self, config: Any, rename_map: dict[str, str]) -> None:
        """Reject a checkpoint whose inputs or action do not fit this environment.

        Environment keys are renamed with ``rename_map``, as for any robot. RGB spatial dimensions are left to
        policy preprocessing, which owns resizing.
        """
        available = {rename_map.get(f.name, f.name): f for f in self.features}
        for name, expected in config.input_features.items():
            # Placeholder cameras come from the checkpoint's own `empty_cameras` setting.
            if name.startswith(_EMPTY_CAMERA_PREFIX) and name.removeprefix(_EMPTY_CAMERA_PREFIX).isdigit():
                continue
            if name not in available:
                raise ValueError(
                    f"Checkpoint feature missing from the simulator (check --rename_map): {name}"
                )
            actual = available[name]
            shape: tuple[int, ...]
            if actual.kind == "rgb":
                compatible = len(expected.shape) == 3 and expected.shape[0] == actual.shape[2]
                shape = (actual.shape[2], actual.shape[0], actual.shape[1])
            else:
                compatible = tuple(expected.shape) == actual.shape
                shape = actual.shape
            if not compatible:
                raise ValueError(
                    f"Checkpoint feature shape mismatch for {name}: expected {expected.shape}, got {shape}"
                )
        if "action" not in config.output_features:
            raise ValueError("Checkpoint must declare the canonical action feature")
        if tuple(config.output_features["action"].shape) != self.action_feature.shape:
            raise ValueError("Checkpoint action dimension differs from the simulator")


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
    elapsed_steps: np.ndarray
