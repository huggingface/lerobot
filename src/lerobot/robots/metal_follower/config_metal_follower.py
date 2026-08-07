#!/usr/bin/env python

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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class MetalFollowerConfigBase:
    """
    Configuration for the Metal arm follower robot: 7 Damiao motors (6 joints + a permanent
    gripper) over classic CAN, driven via the stock `DamiaoMotorsBus` in MIT position control.

    Unlike OpenArms, the gripper is not optional (no `arm_end_type`): the arm always exposes all
    7 motors, all normalized in degrees (the gripper is passed through in raw motor degrees, not
    converted via a stroke table).

    Kept separate from the registered `MetalFollowerConfig` so `BiMetalFollowerConfig` can embed
    per-arm configs without referencing the RobotConfig choice registry (which would make the
    draccus CLI parser tree self-referential).
    """

    # Required; there is no portable default. With can_interface="slcan" this is the adapter's
    # serial port ("/dev/ttyACM0", "/dev/cu.usbmodem1101" on macOS, "COM5" on Windows); with
    # "socketcan" it is an interface name ("can0").
    port: str | None = None

    # "slcan" (any OS, via pyserial) or "socketcan" (Linux only). slcan is the default because
    # it is the only transport that works on all three platforms and needs no privileged setup
    # -- no `slcand`, no `ip link`, no sudo. Measured on a Metal arm over a CANable at 60 Hz, a
    # full follower tick (7 state requests + 7 replies + 7 MIT writes + 7 replies) costs ~5 ms
    # of a 16.7 ms period and loses no frames, so the follower has ample margin either way.
    # Choose "socketcan" on Linux if you want the kernel CAN stack (candump, can-utils, a
    # filtering daemon in front of the bus).
    can_interface: str = "slcan"

    # Metal uses classic CAN @ 1 Mbps (not CAN FD)
    can_bitrate: int = 1_000_000
    use_can_fd: bool = False
    can_data_bitrate: int | None = None

    # Maps motor names to their (send_can_id, recv_can_id) pair. Both arms of a bimanual setup
    # ship with these same ids, which is why each needs its own bus.
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )

    # Soft joint limits (degrees), clipped against on every action. Defaults are the vendor's
    # position_min/max from motor_config.cpp. Narrow them to fence off part of the workspace.
    # The gripper's entry is the vendor's 0..2.4 rad; note its own stroke table only documents
    # jaw opening up to 116.4 deg, so the last stretch is past the widest measured opening.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-160.0, 160.0),
            "shoulder_lift": (-180.0, 0.0),
            "elbow_flex": (0.0, 180.0),
            "wrist_flex": (-123.0, 81.0),
            "wrist_yaw": (-85.0, 85.0),
            "wrist_roll": (-145.0, 145.0),
            "gripper": (0.0, 137.5),
        }
    )

    # Per-motor MIT follow gains {name: (kp, kd)}, applied at connect();
    # Tuned for a smooth follow: high stiffness with heavy damping.
    # If the follow feels mushy rather than smooth, raise velocity_ff_alpha before lowering kd.
    # Mutating the resolved dict at runtime retunes the arm live.
    gains: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (160.0, 6.5),
            "shoulder_lift": (390.0, 11.0),
            "elbow_flex": (320.0, 11.0),
            "wrist_flex": (160.0, 5.0),
            "wrist_yaw": (20.0, 0.6),
            "wrist_roll": (20.0, 0.6),
            "gripper": (20.0, 0.6),
        }
    )

    # Slow initial synchronization: at teleop start the follower can be far from the leader, and
    # firm follow gains would snap it there hard. Until it has caught up, cap each joint's per-step
    # motion to this many degrees (gentle alignment), then track at full speed. None disables it.
    startup_sync_speed_deg: float | None = 1.0
    startup_sync_tolerance_deg: float = 3.0

    # Safety limit for relative target positions (degrees). None disables the check.
    max_relative_target: float | dict[str, float] | None = None

    # Filtered finite-difference velocity feedforward for MIT position control.
    velocity_feedforward: bool = True
    velocity_ff_alpha: float = 0.08
    velocity_ff_max_deg_s: float = 120.0

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = False

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("metal_follower")
@dataclass
class MetalFollowerConfig(RobotConfig, MetalFollowerConfigBase):
    """Registered single-arm metal follower config (adds `id` / `calibration_dir`)."""
