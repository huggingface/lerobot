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

from dataclasses import dataclass, field

from ..config import RobotConfig


@dataclass
class AlmondAxolCameraConfig:
    """One ZED camera slot of the Almond Axol.

    The Axol records through Stereolabs ZED cameras attached to the robot
    machine (opened by serial number via the almond-axol SDK). A camera with
    ``stereo=True`` and ``eyes="both"`` expands into two observation keys,
    ``<name>_left`` / ``<name>_right``, backed by a single grab; ``eyes="left"``
    or ``"right"`` records that single eye under the plain slot name.

    Attributes:
        serial: Serial number of the ZED camera to open.
        fps: Capture rate.
        width: Frame width baked into the dataset features.
        height: Frame height baked into the dataset features.
        stereo: Whether this is a stereo ZED (e.g. ZED X) opened on the stereo
            grab path.
        eyes: Which eye(s) of a stereo camera to record: ``"both"``, ``"left"``
            or ``"right"``.
    """

    serial: int
    fps: int = 60
    width: int = 960
    height: int = 600
    stereo: bool = False
    eyes: str = "both"


@RobotConfig.register_subclass("almond_axol")
@dataclass
class AlmondAxolConfig(RobotConfig):
    """Configuration for the Almond Axol dual-arm robot.

    Attributes:
        left_channel: SocketCAN interface of the left arm.
        right_channel: SocketCAN interface of the right arm.
        telemetry_hz: Background joint-telemetry polling rate in Hz.
        observe_torques: Include joint torques in observations.
        observe_cartesian: Use Cartesian end-effector space for observations and
            actions (per-arm 6-axis pose + gripper) instead of the 16 joint
            positions; actions are resolved back to joint targets via the SDK's
            inverse kinematics.
        cameras: ZED camera slots keyed by name (e.g. ``overhead``,
            ``left_arm``, ``right_arm``).
    """

    left_channel: str = "can_alm_axol_l"
    right_channel: str = "can_alm_axol_r"
    telemetry_hz: float = 120.0
    observe_torques: bool = False
    observe_cartesian: bool = False
    cameras: dict[str, AlmondAxolCameraConfig] = field(default_factory=dict)
