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
from lerobot.transport.wire.protocol import validate_segment


@dataclass(frozen=True)
class EnvDescriptor:
    """Declare canonical observations, controller conventions, and available tasks."""

    sim_type: str
    semantics: str
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    control: str
    fps: float
    clocks: tuple[str, ...]
    tasks: dict[str, list[str]]
    max_episode_steps: int
    sim_build: dict[str, str]
    gripper_indices: tuple[int, ...] = ()
    command_retention_indices: tuple[int, ...] = ()
    # The simulated body, recorded as a sim robot's robot_type; empty when the simulator does not declare one.
    robot_type: str = ""
    # The resolved simulator configuration served, so a client can check it runs the env it expects.
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def hold(self) -> str:
        """Position controllers retain targets; other controllers hold with a neutral command."""
        return "position" if self.control == "position" else "command"

    def __post_init__(self) -> None:
        """Reject invalid declarations before creating simulator resources."""
        if self.control not in {"position", "delta", "eef_pose", "velocity"}:
            raise ValueError("Unknown controller convention")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("FPS and episode limit must be positive")
        if len(self.action_feature.shape) != 1 or not self.action_feature.names:
            raise ValueError("Actions require ordered component names")
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError("Duplicate observation features")
        if any(
            i < 0 or i >= self.action_feature.shape[0]
            for i in self.gripper_indices + self.command_retention_indices
        ):
            raise ValueError("Invalid gripper component")

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit wire fields without Python-object serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvDescriptor":
        """Reconstruct validated declarations from decoded wire fields.

        Fields this version does not know are dropped, so a newer server can add declarations.
        """
        known = {f.name for f in fields(cls)}
        data = {key: value for key, value in data.items() if key in known}
        data["features"] = tuple(FeatureSpec(**f) for f in data["features"])
        data["action_feature"] = FeatureSpec(**data["action_feature"])
        data["clocks"] = tuple(data["clocks"])
        data["gripper_indices"] = tuple(data.get("gripper_indices", ()))
        data["command_retention_indices"] = tuple(data.get("command_retention_indices", ()))
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


@dataclass(frozen=True)
class ExecutionFeedback:
    """An executed control interval and its subsequent observation, not queue admission.

    Applied values are controller commands, not proof of physical motion completion.
    The advanced mask identifies absorbing worlds which did not execute a new action.
    Timestamps use the server monotonic clock, never the client clock.
    """

    request_id: str
    generation: int
    sequence: int
    applied: np.ndarray
    advanced: np.ndarray
    started_at: float
    completed_at: float
    result: StepResult

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata alongside the enclosing reply's single result."""
        # The enclosing reply carries result once, avoiding duplicate image allocation.
        return {name: value for name, value in vars(self).items() if name != "result"}

    @classmethod
    def from_dict(cls, data: dict[str, Any], result: StepResult) -> "ExecutionFeedback":
        """Attach the result carried by the same reply."""
        return cls(**data, result=result)

    def validate(self, descriptor: EnvDescriptor, num_envs: int) -> None:
        """Validate execution identity, applied commands, and their paired observation."""
        validate_segment(self.request_id, "execution request ID")
        if any(
            type(value) is not int or not 0 <= value < 2**63 for value in (self.generation, self.sequence)
        ):
            raise ValueError("Invalid execution identity")
        validate_array(self.applied, descriptor.action_feature, leading_shape=(num_envs,))
        if (
            not isinstance(self.advanced, np.ndarray)
            or self.advanced.shape != (num_envs,)
            or self.advanced.dtype != np.bool_
        ):
            raise ValueError("Invalid execution mask")
        if (
            not np.isfinite([self.started_at, self.completed_at]).all()
            or not 0 <= self.started_at <= self.completed_at
        ):
            raise ValueError("Invalid server execution timestamps")
        validate_result(self.result, descriptor, num_envs)


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
