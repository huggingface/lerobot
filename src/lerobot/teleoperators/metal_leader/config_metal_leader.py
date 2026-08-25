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
class MetalLeaderConfigBase:
    """
    Configuration for the Metal arm leader teleoperator: 7 Damiao motors (6 joints + a permanent
    gripper) over classic CAN, driven via the stock `DamiaoMotorsBus`.

    Unlike torque-disabled leaders, this arm is powered while the human moves it: a background
    thread streams Pinocchio gravity-compensation torque with `kp=0`, so the arm holds its own
    weight and the operator feels only the force needed to accelerate it. The gripper is left
    backdrivable and its raw motor angle drives the follower gripper 1:1.

    Kept separate from the registered `MetalLeaderConfig` so `BiMetalLeaderConfig` can embed
    per-arm configs without referencing the TeleoperatorConfig choice registry (which would make
    the draccus CLI parser tree self-referential).
    """

    # Required; there is no portable default. With can_interface="slcan" this is the adapter's
    # serial port ("/dev/ttyACM0", "/dev/cu.usbmodem1101" on macOS, "COM5" on Windows); with
    # "socketcan" it is an interface name ("can0").
    port: str | None = None

    # "slcan" (any OS, via pyserial) or "socketcan" (Linux only). slcan is the default because
    # it is the only transport available on macOS/Windows and needs no privileged setup. It is
    # latency-bound rather than bandwidth-bound: measured on a Metal arm over a CANable, a full
    # gravity tick (28 frames) takes ~4.8 ms p50 and drops nothing, which is ~48% of the period
    # at the default gravity_hz=100 but exceeds the period outright at 200 -- before Pinocchio's
    # solve is added. `MetalLeader.__init__` warns if you raise gravity_hz above 100 over slcan.
    can_interface: str = "slcan"

    # Metal uses classic CAN @ 1 Mbps (not CAN FD)
    can_bitrate: int = 1_000_000
    use_can_fd: bool = False
    can_data_bitrate: int | None = None

    # Maps motor names to their (send_can_id, recv_can_id) pair. Matches the follower's table:
    # leader and follower are the same arm, so a bimanual pair needs one bus per arm.
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

    # Path to the arm's URDF. Left empty, it is downloaded once from the arm's description
    # repository and cached under `HF_LEROBOT_HOME/metal` (see `urdf.py`). Set it to run offline
    # on a machine that has never fetched it.
    urdf_path: str = ""

    # Gravity-compensation thread rate (Hz). Still an order of magnitude above what the arm's
    # dynamics need — a human arm's bandwidth is under ~10 Hz — so lower it further if the bus
    # gets tight. Each tick costs 28 CAN frames.
    #
    # 100 rather than 200 because a full tick measures ~4.8 ms p50 over the default slcan
    # transport, which fits inside this 10 ms period but not inside 200 Hz's 5 ms. On socketcan
    # (~3.9 ms p50) 200 Hz is feasible and costs roughly 62% of a 1 Mbps bus.
    gravity_hz: int = 100

    # MIT damping gain while gravity-compensated. kp is always 0 so the human can position the arm
    # freely; kd supplies velocity damping. kd is also the brake against friction-feedforward
    # runaway — don't drive it to 0 while raising friction_scale unless you have tested stability
    # at your control rate. Accepts a single float (all joints) or a per-joint dict; motors absent
    # from the dict get 0. The vendor uses a uniform kd=0 with full friction feedforward, and gets
    # its per-joint feel from the viscous coefficients instead.
    leader_kd: float | dict[str, float] = 0.0

    # Friction/Coriolis feedforward: fed the measured joint velocity to cancel the arm's own
    # gearbox friction so the leader feels transparent. Per-joint because this arm's real friction
    # differs from the vendor's viscous coefficients; a uniform 1.0 did not fit. Higher is lighter,
    # but past a joint-specific threshold the joint RUNS AWAY. 0 leaves that joint gravity-only.
    use_velocity_feedforward: bool = True
    friction_scale: float | dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 1.4,
            "shoulder_lift": 3.3,
            "elbow_flex": 1.1,
            "wrist_flex": 0.7,
            "wrist_yaw": 0.3,
            "wrist_roll": 0.7,
        }
    )

    # Measured velocity below this counts as zero. Motor velocity noise would otherwise make the
    # friction term chatter while the arm is held still.
    velocity_deadzone_rad_s: float = 0.05

    # Scales the gripper's own friction feedforward so the jaws are easy to squeeze (the gripper
    # is not in the URDF dynamics model). 0 leaves the gripper at zero torque.
    gripper_friction_scale: float = 1.0

    # On disconnect the gravity thread stops, so the arm would fall on the last zero-kp command.
    # Freeze it in place with these gains instead: it stays up, no longer weightless. Set
    # hold_kp_on_disconnect=0 to leave it limp and backdrivable.
    hold_kp_on_disconnect: float = 50.0
    hold_kd_on_disconnect: float = 1.0


@TeleoperatorConfig.register_subclass("metal_leader")
@dataclass
class MetalLeaderConfig(TeleoperatorConfig, MetalLeaderConfigBase):
    """Registered single-arm metal leader config (adds `id` / `calibration_dir`)."""
