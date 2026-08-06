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

_GAINS: dict[str, dict[str, list[float]]] = {
    "left_leg": {
        "kp": [150, 150, 150, 300, 40, 40],
        "kd": [2, 2, 2, 4, 2, 2],
    },  # pitch, roll, yaw, knee, ankle_pitch, ankle_roll
    "right_leg": {"kp": [150, 150, 150, 300, 40, 40], "kd": [2, 2, 2, 4, 2, 2]},
    "waist": {"kp": [250, 250, 250], "kd": [5, 5, 5]},  # yaw, roll, pitch
    "left_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},  # shoulder_pitch/roll/yaw, elbow
    "left_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},  # roll, pitch, yaw
    "right_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},
    "right_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},
}


def _build_gains() -> tuple[list[float], list[float]]:
    """Build kp and kd lists from body-part groupings."""
    kp = [v for g in _GAINS.values() for v in g["kp"]]
    kd = [v for g in _GAINS.values() for v in g["kd"]]
    return kp, kd


_DEFAULT_KP, _DEFAULT_KD = _build_gains()


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    """Configuration for the Unitree G1 humanoid.

    The G1 is reached over a ZMQ bridge rather than a serial bus, so there is no `port` field and
    calibration is handled by the robot's own firmware.

    All 29 joints are addressed by index, so `kp`, `kd` and `default_positions` are lists in the G1's joint
    order: left leg, right leg, waist, left arm, left wrist, right arm, right wrist.

    Args:
        kp (`list[float]`, *optional*):
            Per-joint proportional gains, 29 values. Defaults to the per-body-part gains recommended by
            Unitree.
        kd (`list[float]`, *optional*):
            Per-joint derivative gains, 29 values.
        default_positions (`list[float]`, *optional*):
            Per-joint home positions, 29 values. Defaults to all zeros.
        control_dt (`float`, *optional*, defaults to 0.004):
            Control loop timestep in seconds, i.e. 250 Hz.
        is_simulation (`bool`, *optional*, defaults to `True`):
            Whether to drive a MuJoCo simulation instead of the physical robot. Keep `True` until the
            behaviour is validated in sim.
        robot_ip (`str`, *optional*, defaults to `"192.168.123.164"`):
            Address of the robot's ZMQ bridge. The default is the G1's factory address.
        cameras (`dict[str, CameraConfig]`, *optional*):
            ZMQ-based remote cameras to read alongside the joint states.
        gravity_compensation (`bool`, *optional*, defaults to `False`):
            Whether to compensate for gravity on the arms using the arm IK solver.
        controller (`str`, *optional*):
            Class name of the lower-body locomotion controller, e.g. `"GrootLocomotionController"` or
            `"HolosomaLocomotionController"`. `None` leaves the legs uncontrolled.
        id (`str`, *optional*):
            Identifier for this particular robot.
        calibration_dir (`Path`, *optional*):
            Unused: the G1 manages its own calibration.
    """

    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # Default joint positions
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Launch mujoco simulation
    is_simulation: bool = True

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP

    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Lower-body controller class name, e.g. "GrootLocomotionController" or
    # "HolosomaLocomotionController". None disables it.
    controller: str | None = None
