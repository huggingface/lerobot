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

"""Env metadata and step results, independent of how the env is served."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from lerobot.utils.feature_spec import FeatureSpec

# What an action commands: joint targets, end-effector deltas, or end-effector poses.
CONTROL_MODES = ("position", "delta", "eef_pose")


@dataclass(frozen=True)
class EnvDescriptor:
    """What an env exposes: observation and action features, control mode, fps and tasks.

    Args:
        sim_type (`str`):
            Integration that runs the env, e.g. `libero`.
        semantics (`str`):
            Name of the observation and action conventions, e.g. `libero-delta-v1`.
        features (`tuple[FeatureSpec, ...]`):
            Observation features, with LeRobot keys.
        action_feature (`FeatureSpec`):
            The action: one dimension, with ordered component names.
        control_mode (`str`):
            What an action commands: `position`, `delta` or `eef_pose`.
        fps (`float`):
            Control rate.
        tasks (`dict[str, list[int]]`):
            Task IDs of each suite.
        max_episode_steps (`int`):
            Steps before an episode is truncated.
        robot_type (`str`, *optional*, defaults to `""`):
            Body a sim robot records in its dataset.
        config (`dict[str, Any]`, *optional*):
            The `SimConfig` the env was built with. A client checks it against its own settings.
    """

    sim_type: str
    semantics: str
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    control_mode: str
    fps: float
    tasks: dict[str, list[int]]
    max_episode_steps: int
    robot_type: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid environment metadata."""
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode {self.control_mode!r}; expected one of {CONTROL_MODES}")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("FPS and episode limit must be positive")
        if len(self.action_feature.shape) != 1 or not self.action_feature.names:
            raise ValueError("Actions require ordered component names")
        if len({feature.name for feature in self.features}) != len(self.features):
            raise ValueError("Duplicate observation features")

    def observation_space(self) -> spaces.Dict:
        """Return the observation space of one env: RGB images as uint8, other features unbounded."""
        return spaces.Dict(
            {
                f.name: spaces.Box(0, 255, f.shape, np.uint8)
                if f.kind == "rgb"
                else spaces.Box(-np.inf, np.inf, f.shape, np.dtype(f.dtype))
                for f in self.features
            }
        )

    def action_space(self) -> spaces.Box:
        """Return the action space of one env: unbounded float32."""
        return spaces.Box(-np.inf, np.inf, self.action_feature.shape, np.float32)

    def same_schema(self, other: "EnvDescriptor") -> bool:
        """Return whether `other` has the same observation features, action feature and semantics."""
        return (
            self.features == other.features
            and self.action_feature == other.action_feature
            and self.semantics == other.semantics
        )


@dataclass(frozen=True)
class StepResult:
    """Result of one reset or step of a batch of envs. Arrays have the batch dimension first.

    Args:
        obs (`dict[str, np.ndarray]`):
            Observations, with LeRobot keys. A finished env keeps its terminal observation.
        task (`tuple[str, ...]`):
            Task description of each env.
        reward (`np.ndarray`):
            Reward of each env, `float32`. Zero once its episode has finished.
        terminated (`np.ndarray`):
            Whether each env reached a terminal state, `bool`.
        truncated (`np.ndarray`):
            Whether each env's episode was cut short, e.g. by its step limit, `bool`.
        is_success (`np.ndarray`):
            Whether each env completed its task, `bool`.
        elapsed_steps (`np.ndarray`):
            Steps since the last reset, `int64`.
        metrics (`dict[str, np.ndarray]`, *optional*, defaults to `{}`):
            What the simulator reports besides success, one `float32` value per env, e.g. a progress score.
            An integration fills it from its own step info. LeRobot records the last value of each episode
            and never aggregates it, so a simulator chooses its own names and units.
    """

    obs: dict[str, np.ndarray]
    task: tuple[str, ...]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    is_success: np.ndarray
    elapsed_steps: np.ndarray
    metrics: dict[str, np.ndarray] = field(default_factory=dict)

    def frames(self) -> tuple[np.ndarray, ...]:
        """Return one frame per env from the first RGB observation, including terminal frames."""
        key = next((k for k, v in self.obs.items() if v.dtype == np.uint8 and v.ndim == 4), None)
        if key is None:
            raise ValueError("This environment exposes no RGB observation")
        return tuple(frame.copy() for frame in self.obs[key])

    def info(self) -> dict[str, Any]:
        """Return the Gymnasium `info` dict for this step."""
        return {
            "is_success": self.is_success,
            "task": self.task,
            "elapsed_steps": self.elapsed_steps,
            "metrics": self.metrics,
        }
