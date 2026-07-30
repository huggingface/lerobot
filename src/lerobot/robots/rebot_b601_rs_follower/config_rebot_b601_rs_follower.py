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

    The B601-RS is the 6-DOF + gripper reBot B601 chassis driven by RobStride RS
    CAN motors (the ``rs-06`` base joints and ``rs-00`` wrists/gripper).
    RobStride motors are MIT-mode only; motor communication goes through the
    ``motorbridge`` package via ``add_robstride_motor`` over a CAN bus, reached
    either through a SocketCAN interface (the default) or a Damiao USB-to-CAN
    serial bridge (the bridge is transport-agnostic and can carry RobStride
    frames too). There is no POS_VEL / FORCE_POS path (those are Damiao-only).
    """

    # Communication port. For ``can_adapter="socketcan"`` this is the CAN channel
    # name (e.g. "can0"); for ``can_adapter="damiao"`` it is the Damiao USB-to-CAN
    # serial bridge device (e.g. "/dev/ttyACM0").
    port: str

    # CAN transport. RobStride motors talk CAN directly, so either transport works:
    #   "socketcan" - SocketCAN channel (e.g. "can0"); the default.
    #   "damiao"    - Damiao USB-to-CAN serial bridge (the bridge is transport-
    #                  agnostic; it can carry RobStride CAN frames too).
    can_adapter: str = "socketcan"

    # Baud rate for the Damiao serial bridge (only used when can_adapter="damiao").
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety purposes (in degrees). Set to a positive scalar to apply the
    # same value to all motors, or to a dict mapping motor names to per-motor values.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps motor names to their (send_can_id, recv_can_id) pair. RobStride motors
    # share a single feedback CAN id (0xFD); send ids stay 1..7.
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0xFD),
            "shoulder_lift": (0x02, 0xFD),
            "elbow_flex": (0x03, 0xFD),
            "wrist_flex": (0x04, 0xFD),
            "wrist_yaw": (0x05, 0xFD),
            "wrist_roll": (0x06, 0xFD),
            "gripper": (0x07, 0xFD),
        }
    )

    # MIT kp/kd per arm joint (motor order: shoulder_pan..gripper). The gripper
    # entry is unused — the gripper is driven by the impedance torque below.
    mit_kp: float | list[float] = field(
        default_factory=lambda: [50.0, 150.0, 150.0, 50.0, 50.0, 50.0, 12.0]
    )
    mit_kd: float | list[float] = field(
        default_factory=lambda: [3.0, 10.0, 10.0, 5.0, 4.0, 4.0, 0.05]
    )

    # MIT impedance gripper: Kp/Kd of the external impedance torque
    #   tau = Kp*(pos_target - pos) + Kd*(target_vel - vel),
    # clamped to the torque limits below. The motor itself is driven purely by
    # this feedforward torque (kp=0, plus a small fixed damping), which bounds
    # grip force instead of pushing to the position target regardless of force.
    gripper_mit_kp: float = 12.0
    gripper_mit_kd: float = 0.05

    # MIT impedance gripper: max |feedforward torque| (N·m) while moving.
    gripper_mit_torque_limit: float = 3.5
    # MIT impedance gripper: max |feedforward torque| (N·m) at near-zero speed —
    # a gentler grasp/hold limit so the gripper doesn't crush or overcurrent.
    gripper_mit_hold_torque_limit: float = 1.0

    # Physical motor travel ranges (degrees) — the RobStride motors' actual
    # movement. RobStride motors are mounted opposite to the Damiao variant, so
    # these are the positive-physical ranges (matching the Seeed RS reference).
    # The incoming (Damiao-convention) action is flipped into them via
    # `joint_directions` before clipping (see `send_action`).
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-145.0, 145.0),
            "shoulder_lift": (-0.0, 170.0),
            "elbow_flex": (-0.0, 200.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-0.0, 270.0),
        }
    )

    # Per-joint direction applied to the incoming action BEFORE clipping, mapping
    # the shared reBot Arm 102 leader's Damiao-convention output into the
    # RobStride motor's positive-physical range. This is the composite of the
    # leader's `joint_directions` and the Seeed RS `joint_directions` — which is
    # -1 for every joint (the leader already applies the gripper ×6 scale, so
    # here the gripper only carries the sign flip).
    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": -1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": -1.0,
            "wrist_flex": -1.0,
            "wrist_yaw": -1.0,
            "wrist_roll": -1.0,
            "gripper": -1.0,
        }
    )


@RobotConfig.register_subclass("rebot_b601_rs_follower")
@dataclass
class RebotB601RSFollowerRobotConfig(RobotConfig, RebotB601RSFollowerConfig):
    """Registered configuration for the reBot B601-RS follower robot."""

    pass
