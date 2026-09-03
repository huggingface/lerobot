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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class RebotB601RSFollowerConfig:
    """Base configuration class for the Seeed Studio reBot B601-RS follower arm.

    The B601-RS is the Robstride build of the reBot B601: a 6-DOF arm plus gripper
    driven by Robstride RS-series CAN motors speaking their factory-default
    ("private") protocol. Motor communication goes through
    :class:`~lerobot.motors.robstride.RobstridePrivateMotorsBus` (python-can).
    """

    # CAN channel: "can0" for SocketCAN adapters, "/dev/tty..." for slcan bridges.
    port: str = "can0"

    # CAN interface type: "auto" (default), "socketcan" or "slcan". Auto selects
    # slcan for "/dev/..." ports, socketcan otherwise.
    can_interface: str = "auto"

    # Host id placed in outgoing frames and expected in replies. 0xFD matches the
    # vendor tools; only one bus master may use the bus at a time.
    host_id: int = 0xFD

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety purposes (in degrees). Set to a positive scalar to apply the
    # same value to all motors, or to a dict mapping motor names to per-motor values.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps motor names to their CAN device id. The private protocol has no separate
    # receive id: replies are addressed to `host_id`.
    motor_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_yaw": 5,
            "wrist_roll": 6,
            "gripper": 7,
        }
    )

    # Maps motor names to their RS-series model ("rs00".."rs06"), which selects the
    # MIT frame packing and feedback scalings.
    motor_models: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "rs06",
            "shoulder_lift": "rs06",
            "elbow_flex": "rs06",
            "wrist_flex": "rs00",
            "wrist_yaw": "rs00",
            "wrist_roll": "rs00",
            "gripper": "rs00",
        }
    )

    # Arm control: "position" (on-motor profile position mode) or "mit".
    control_mode: str = "position"

    # Max speed (deg/s) per joint in Position mode (motor order), written to the
    # motors' `limit_spd` parameter. Unused when control_mode="mit".
    position_speed_limit: float | list[float] = field(
        default_factory=lambda: [150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 900.0]
    )

    # MIT kp/kd per arm joint (motor order), tuned on a physical B601-RS. Unused when
    # control_mode="position". Note the RS MIT gain full-scales are model-dependent
    # (rs06 packs kp over 0..5000, rs00 over 0..500).
    mit_kp: float | list[float] = field(default_factory=lambda: [50.0, 150.0, 150.0, 50.0, 50.0, 50.0, 50.0])
    mit_kd: float | list[float] = field(default_factory=lambda: [3.0, 10.0, 10.0, 5.0, 4.0, 4.0, 4.0])

    # MIT only: gripper gains. Robstride motors have no FORCE_POS equivalent, so in
    # MIT mode the gripper is driven like the arm joints, with these gains.
    gripper_mit_kp: float = 50.0
    gripper_mit_kd: float = 4.0

    # Soft joint limits (degrees). These are clipped against on every action.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-150.0, 150.0),
            "shoulder_lift": (-200.0, 1.0),
            "elbow_flex": (-200.0, 1.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )


@RobotConfig.register_subclass("rebot_b601_rs_follower")
@dataclass
class RebotB601RSFollowerRobotConfig(RobotConfig, RebotB601RSFollowerConfig):
    """Registered configuration for the reBot B601-RS follower robot."""

    pass
