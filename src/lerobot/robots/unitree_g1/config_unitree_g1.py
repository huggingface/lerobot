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
from math import isfinite

from lerobot.cameras import CameraConfig
from lerobot.envs.configs import G1EndEffector, UnitreeG1MujocoEnv

from ..config import RobotConfig
from .g1_embodiments import get_g1_embodiment
from .g1_utils import NUM_MOTORS

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


def _g1_23_gains() -> tuple[list[float], list[float]]:
    # Derived from Unitree's G1_23_ArmController; see the G1 embodiment documentation.
    spec = get_g1_embodiment("g1_23")
    kp, kd = [0.0] * NUM_MOTORS, [0.0] * NUM_MOTORS
    weak_joints = {spec.joint_index.kLeftAnklePitch, spec.joint_index.kRightAnklePitch}
    arm_indices = {joint.value for joint in spec.arm_index}
    for joint in spec.joint_index:
        if "Wrist" in joint.name:
            kp[joint.value], kd[joint.value] = 40.0, 1.5
        elif joint.value in arm_indices or joint in weak_joints:
            kp[joint.value], kd[joint.value] = 80.0, 3.0
        else:
            kp[joint.value], kd[joint.value] = 300.0, 3.0
    return kp, kd


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    kp: list[float] | None = None
    kd: list[float] | None = None

    # Default joint positions
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Launch mujoco simulation
    is_simulation: bool = True

    # Supports dummy, dex1, dex3
    end_effector: G1EndEffector = G1EndEffector.DEX1

    # Loads the lerobot/unitree-g1-mujoco environment
    sim_env: UnitreeG1MujocoEnv = field(init=False)

    # Where the sim's cameras are published, or its viewer instead when publishing is off.
    sim_publish_images: bool = True
    sim_camera_port: int = 5555

    # Toggle the viewer on or off
    sim_onscreen: bool | None = None

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP

    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Controller class name, e.g. GrootLocomotionController / HolosomaLocomotionController /
    # SonicWholeBodyController. None disables it.
    controller: str | None = None

    embodiment: str = "g1_29"

    def __post_init__(self):
        super().__post_init__()
        spec = get_g1_embodiment(self.embodiment)
        if self.controller is not None and not spec.supports_controller:
            raise ValueError(f"Controllers are not supported for {self.embodiment}")
        if self.gravity_compensation and not spec.supports_gravity_compensation:
            raise ValueError(f"Gravity compensation is not implemented for {self.embodiment}")
        default_kp, default_kd = (_DEFAULT_KP, _DEFAULT_KD) if self.embodiment == "g1_29" else _g1_23_gains()
        self.kp = list(default_kp if self.kp is None else self.kp)
        self.kd = list(default_kd if self.kd is None else self.kd)
        inactive = set(range(NUM_MOTORS)) - {joint.value for joint in spec.joint_index}
        for name in ("kp", "kd", "default_positions"):
            values = getattr(self, name)
            if len(values) != NUM_MOTORS or not all(isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {NUM_MOTORS} finite values in DDS slot order")
            if name in ("kp", "kd") and any(value < 0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
            if any(values[index] != 0 for index in inactive):
                raise ValueError(f"{name} must be zero at inactive {self.embodiment} DDS slots")
        self.end_effector = G1EndEffector(self.end_effector)  # from Python it is still a string
        self.sim_env = UnitreeG1MujocoEnv(
            publish_images=self.sim_publish_images,
            camera_port=self.sim_camera_port,
            onscreen=self.sim_onscreen,
            end_effector=self.end_effector,
        )
