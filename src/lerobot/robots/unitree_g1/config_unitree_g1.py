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
from .g1_utils import G1_MOTOR_MODELS, STIFF_JOINT_INDICES, compute_pd_gains

# Default PD gains are derived from Unitree motor physics (armature + target natural
# frequency; see g1_utils.compute_pd_gains) rather than hand-tuned. Controllers may tweak
# individual joints on top of these.
_kp, _kd = compute_pd_gains(G1_MOTOR_MODELS, STIFF_JOINT_INDICES)
_DEFAULT_KP: list[float] = _kp.tolist()
_DEFAULT_KD: list[float] = _kd.tolist()


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
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

    # Controller class name, e.g. GrootLocomotionController / HolosomaLocomotionController /
    # SonicWholeBodyController. None disables it.
    controller: str | None = None
