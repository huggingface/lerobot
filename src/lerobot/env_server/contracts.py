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

"""Canonical simulator descriptors and transitions."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from lerobot.transport.wire.features import FeatureSpec, validate_array


@dataclass(frozen=True)
class EnvDescriptor:
    """Declare canonical observations, controller conventions, and available tasks."""

    sim_type: str
    semantics: str
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    control: str
    fps: float
    tasks: dict[str, list[str]]
    max_episode_steps: int | None
    sim_build: dict[str, str]
    # Empty when the provider does not declare the simulated robot.
    robot_type: str = ""
    # Used to check that the server runs the requested environment.
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate controller and feature declarations."""
        if self.control not in {"position", "delta", "eef_pose", "velocity"}:
            raise ValueError("Unknown controller convention")
        if (
            not np.isfinite(self.fps)
            or self.fps <= 0
            or (self.max_episode_steps is not None and self.max_episode_steps <= 0)
        ):
            raise ValueError("FPS and episode limit must be positive")
        if len(self.action_feature.shape) != 1 or not self.action_feature.names:
            raise ValueError("Actions require ordered component names")
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError("Duplicate observation features")
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in self.sim_build.items()
        ):
            raise ValueError("Simulator versions must map strings to strings")

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit wire fields without Python-object serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvDescriptor":
        """Reconstruct validated declarations, ignoring fields added by a newer server."""
        known = {f.name for f in fields(cls)}
        data = {key: value for key, value in data.items() if key in known}
        data["features"] = tuple(FeatureSpec(**f) for f in data["features"])
        data["action_feature"] = FeatureSpec(**data["action_feature"])
        return cls(**data)


@dataclass(frozen=True)
class StepResult:
    """Carry a batched canonical transition with simulator time and episode status."""

    obs: dict[str, np.ndarray]
    task: tuple[str, ...]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    is_success: np.ndarray
    sim_time: float
    step: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit wire fields without Python-object serialization."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepResult":
        """Reconstruct validated declarations from decoded wire fields."""
        # JPEG replies carry one frame per world; stack them back into the batch.
        obs = {
            name: np.stack(value) if isinstance(value, list) else value for name, value in data["obs"].items()
        }
        return cls(**{**data, "obs": obs, "task": tuple(data["task"])})


def validate_result(result: StepResult, descriptor: EnvDescriptor, num_envs: int) -> None:
    """Reject transitions that disagree with the negotiated schema or batch size."""
    if set(result.obs) != {f.name for f in descriptor.features}:
        raise ValueError("Simulator observation keys differ from the negotiated schema")
    for feature in descriptor.features:
        validate_array(result.obs[feature.name], feature, leading_shape=(num_envs,))
    if len(result.task) != num_envs or any(not isinstance(task, str) for task in result.task):
        raise ValueError("Simulator task descriptions must match the batch")
    for name in ("reward", "terminated", "truncated", "is_success", "step"):
        array = getattr(result, name)
        if not isinstance(array, np.ndarray) or array.shape != (num_envs,):
            raise ValueError(f"Simulator transition batch mismatch: {name}")
        expected = (
            "bool"
            if name in {"terminated", "truncated", "is_success"}
            else "int64"
            if name == "step"
            else "float32"
        )
        if array.dtype.name != expected:
            raise ValueError(f"Simulator transition dtype mismatch: {name}")
    if (
        not np.isfinite(result.reward).all()
        or not np.isfinite(result.sim_time)
        or result.sim_time < 0
        or (result.step < 0).any()
    ):
        raise ValueError("Invalid simulator reward/time/step values")
