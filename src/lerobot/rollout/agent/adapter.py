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

"""Describe LeRobot position commands to the upstream native-tool agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _inspect_robots_agent_available, require_package

from .configuration import AgentSettings
from .kinematics import CartesianKinematics

if TYPE_CHECKING or _inspect_robots_agent_available:
    from inspect_robots.embodiment import EmbodimentInfo
    from inspect_robots.spaces import (
        ActionSemantics,
        Box,
        CameraSpec,
        ObservationSpace,
        StateField,
        StateSpec,
    )
    from inspect_robots.types import Observation


class RobotAdapter:
    def __init__(self, config: AgentSettings, keys: list[str], features: dict, robot_type: str, fps: float):
        require_package("inspect-robots-agent", extra="agent", import_name="inspect_robots_agent")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Agent command fps must be finite and positive")
        self.config, self.keys, self.fps = config, keys, fps
        if not config.robot_notes.strip():
            raise ValueError(
                "inference.robot_notes must describe the workspace, axes, cameras and gripper polarity"
            )
        if not keys or set(keys) != set(config.axes) or any(not k.endswith(".pos") for k in keys):
            raise ValueError(
                "Agent control requires an explicit axis contract for EVERY robot position action; velocity/torque actions are unsupported"
            )
        if any(features.get(k) is not float for k in keys):
            raise ValueError("Each commanded position needs a measured scalar observation of the same name")
        self.low = np.array([config.axes[k].minimum for k in keys])
        self.high = np.array([config.axes[k].maximum for k in keys])
        self.step = np.array([config.axes[k].max_speed / fps for k in keys])
        self.measurement_tolerance = np.array([config.axes[k].measurement_tolerance for k in keys])
        self.tolerance = np.array([config.axes[k].tracking_tolerance for k in keys])
        self.cameras = {k: v for k, v in features.items() if isinstance(v, tuple)}
        if not self.cameras or any(len(shape) != 3 or shape[2] != 3 for shape in self.cameras.values()):
            raise ValueError("Agent control requires at least one RGB HWC camera")
        self.arms = {}
        used: set[str] = set()
        for name, arm in config.cartesian.items():
            arm_keys = set(arm.joints) | ({arm.gripper} if arm.gripper else set())
            if used & arm_keys or not arm_keys <= set(keys) or arm.gripper in arm.joints:
                raise ValueError("Cartesian arms must map distinct robot actions")
            used |= arm_keys
            self.arms[name] = CartesianKinematics(arm, config.axes)
        if self.arms and used != set(keys):
            raise ValueError(
                "Cartesian control must cover every action (including grippers); use joint control otherwise"
            )
        self.labels: list[str] = []
        lows, highs, speeds = [], [], []
        notes = [config.robot_notes, "Native measured/action axes:"]
        for k, axis in config.axes.items():
            notes.append(
                f"{k}: {axis.unit}; {axis.description}; bounds [{axis.minimum}, {axis.maximum}]; maximum speed {axis.max_speed} {axis.unit}/s."
            )
        if self.arms:
            notes.append(
                "Cartesian XYZ are metres in EACH arm's configured base frame. Orientation is relative to its measured orientation at trial start: R = Rz(yaw) Ry(-pitch) Rx(roll) R_start. Positive pitch is about NEGATIVE base Y; all angles radians. Pinned angles cannot be changed. These frames do not imply a camera-to-base calibration. Do not infer metric pixel positions without calibrated geometry. Cartesian waypoints are solved locally with IK; no collision checking is provided."
            )
            for name, kin in self.arms.items():
                arm = kin.config
                notes.append(f"{name}: {arm.frame_description}")
                self.labels.extend(f"{name}_{suffix}" for suffix in ("x", "y", "z", "yaw", "pitch", "roll"))
                lows.extend(arm.position_low + arm.angle_low)
                highs.extend(arm.position_high + arm.angle_high)
                speeds.extend([arm.linear_speed] * 3 + [arm.angular_speed] * 3)
                if arm.gripper:
                    axis = config.axes[arm.gripper]
                    self.labels.append(f"{name}_gripper")
                    lows.append(axis.minimum)
                    highs.append(axis.maximum)
                    speeds.append(axis.max_speed)
                    notes.append(f"{name}_gripper maps to {arm.gripper} in its native units.")
        else:
            self.labels = list(keys)
            lows, highs, speeds = self.low.tolist(), self.high.tolist(), (self.step * fps).tolist()
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Agent dimension names must be unique")
        self.command_low, self.command_high = np.array(lows), np.array(highs)
        shape = (len(self.labels),)
        self.info = EmbodimentInfo(
            name=robot_type,
            control_hz=fps,
            docs="\n".join(notes),
            action_space=Box(
                shape,
                self.command_low,
                self.command_high,
                ActionSemantics(
                    control_mode="eef_abs_pose" if self.arms else "joint_pos",
                    rotation_repr="euler_xyz" if self.arms else "none",
                    gripper="continuous",
                    dim_labels=tuple(self.labels),
                    max_step=tuple(
                        speed / fps if lo != hi else None
                        for speed, lo, hi in zip(speeds, lows, highs, strict=True)
                    ),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(name=k, height=v[0], width=v[1], channels=3) for k, v in self.cameras.items()
                ),
                state=StateSpec(
                    (StateField("command_state", shape, "per-dimension units in embodiment notes"),)
                ),
            ),
        )
        self.pose: dict = {}

    def vector(self, pose: dict, *, feedback: bool = True) -> np.ndarray:
        vector = np.array([pose[k] for k in self.keys], dtype=float)
        margin = self.measurement_tolerance if feedback else 1e-7
        if (
            not np.isfinite(vector).all()
            or np.any(vector < self.low - margin)
            or np.any(vector > self.high + margin)
        ):
            raise ValueError("Measured robot position is outside the declared limits")
        return vector if feedback else np.clip(vector, self.low, self.high)

    def reset(self, pose: dict):
        for kin in self.arms.values():
            kin.reset(pose)

    def observation(self, raw: dict, task: str, steps: int, feedback: list, approvals: list) -> Observation:
        self.vector(raw)
        self.pose = {k: float(raw[k]) for k in self.keys}
        if self.arms:
            state = []
            for kin in self.arms.values():
                state.extend(kin.observe(raw))
                if kin.config.gripper:
                    state.append(float(raw[kin.config.gripper]))
            values = np.array(state)
            # Report actual measured orientation, including residuals on pinned target
            # dimensions. Upstream clips command targets; observations must stay truthful.
        else:
            values = self.vector(raw)
        images = {}
        for name, shape in self.cameras.items():
            frame = np.asarray(raw[name])
            if frame.shape != shape or frame.dtype != np.uint8:
                raise ValueError(f"{name}: expected uint8 RGB image with shape {shape}")
            images[name] = frame.copy()
        return Observation(
            images=images,
            state={"command_state": values},
            instruction=task,
            extra={"env_step": steps, "operator_messages": feedback, "approvals": approvals},
        )

    def translate(self, waypoints: np.ndarray) -> list[np.ndarray]:
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != len(self.labels) or not np.isfinite(waypoints).all():
            raise ValueError("Malformed motion waypoints")
        if np.any(waypoints < self.command_low - 1e-7) or np.any(waypoints > self.command_high + 1e-7):
            raise ValueError("Motion exceeds configured command bounds")
        pose = self.pose.copy()
        previous = self.vector(pose)
        result = []
        for point in waypoints:
            if self.arms:
                offset = 0
                for kin in self.arms.values():
                    pose.update(kin.solve(point[offset : offset + 6], pose))
                    offset += 6
                    if kin.config.gripper:
                        pose[kin.config.gripper] = float(point[offset])
                        offset += 1
                current = self.vector(pose, feedback=False)
            else:
                current = point.copy()
            if np.any(np.abs(current - previous) > self.step + 1e-7):
                raise ValueError(
                    "Motion exceeds native joint/gripper speed; request a smaller/slower reachable move"
                )
            result.append(current)
            previous = current
        return result

    def pre_check(self, waypoints: np.ndarray) -> str | None:
        try:
            self.translate(waypoints)
        except ValueError as exc:
            return str(exc)
        return None
