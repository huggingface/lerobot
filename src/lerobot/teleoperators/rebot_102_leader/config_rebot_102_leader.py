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

from ..config import TeleoperatorConfig


@dataclass
class RebotArm102LeaderConfig:
    """Base configuration class for the Seeed Studio StarArm102 / reBot Arm 102 leader.

    The reBot Arm 102 is a 7-joint (incl. gripper) leader arm driven by FashionStar
    UART smart servos. Servo communication goes through ``motorbridge-smart-servo``.
    """

    # USB-to-UART device the leader arm is connected to (e.g. "/dev/ttyUSB0").
    port: str

    baudrate: int = 1_000_000

    # Servo id of each joint on the UART bus.
    joint_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_yaw": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
    )

    # Per-joint signed scale applied to raw servo angles so the leader matches the follower
    # convention: output = raw * direction. The sign flips direction; the magnitude rescales
    # a leader joint's travel onto a follower joint with a different range (e.g. the gripper
    # uses -6 to widen onto the reBot B601 gripper; a follower joint with a smaller range than
    # the leader uses a fractional magnitude like -0.667). Float so non-integer scales work.
    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": -1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
            "wrist_yaw": 1.0,
            "wrist_roll": -1.0,
            "gripper": -6.0,
        }
    )

    # Per-joint [min, max] output range in degrees. Matches the reBot B601 follower
    # joint limits so leader actions can drive the follower key-for-key.
    joint_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "shoulder_pan": [-150, 150],
            "shoulder_lift": [-200, 1],
            "elbow_flex": [-200, 1],
            "wrist_flex": [-80, 90],
            "wrist_yaw": [-90, 90],
            "wrist_roll": [-90, 90],
            "gripper": [-270, 0],
        }
    )


@TeleoperatorConfig.register_subclass("rebot_102_leader")
@dataclass
class RebotArm102LeaderTeleopConfig(TeleoperatorConfig, RebotArm102LeaderConfig):
    """Registered configuration for the reBot Arm 102 leader teleoperator."""

    pass
