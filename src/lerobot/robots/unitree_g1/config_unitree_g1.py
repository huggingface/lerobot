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

# Rest / soft-stop arm pose. The G1 elbow's mechanical zero sits ~90deg (forearm
# pointing forward); a positive elbow angle *extends* the arm toward straight (this is
# why holosoma uses 0.6 for a mildly-extended natural stance). We hang the arms nearly
# straight down so that on soft-stop they're already down and don't drop as dead weight
# when the joints go passive. If the arms curl the wrong way on your robot, flip the
# sign of _REST_ELBOW.
_LEFT_ELBOW_IDX = 18
_RIGHT_ELBOW_IDX = 25
_REST_ELBOW = 1.17  # rad, ~160deg forearm (0 rad~=90deg, 1.5 rad~=180deg straight)


def _build_default_positions() -> list[float]:
    pos = [0.0] * 29
    pos[_LEFT_ELBOW_IDX] = _REST_ELBOW
    pos[_RIGHT_ELBOW_IDX] = _REST_ELBOW
    return pos


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # Default joint positions (rest / soft-stop pose; arms hang straight down)
    default_positions: list[float] = field(default_factory=_build_default_positions)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Launch mujoco simulation
    is_simulation: bool = True

    # Run the locomotion controller ONBOARD the robot (policy on the G1 itself,
    # against local DDS) instead of on the laptop over the ZMQ bridge. In this mode
    # the robot object uses the real Unitree SDK channels and expects high-level
    # actions (arm targets + joystick axes) to be fed in via send_action (e.g. by
    # run_g1_onboard.py, which receives them from the laptop). Mutually exclusive
    # with is_simulation.
    onboard: bool = False
    # DDS network interface for onboard mode (None = SDK default, matching
    # run_g1_server.py's ChannelFactoryInitialize(0)).
    dds_interface: str | None = None

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP

    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Lower-body controller class name, e.g. "GrootLocomotionController" or
    # "HolosomaLocomotionController". None disables it.
    controller: str | None = None

    # On disconnect, ramp the arms slowly back to `default_positions` (hands down)
    # before going passive, instead of dropping straight to zero torque. Only
    # applies on the real robot when a locomotion controller holds the legs.
    soft_stop: bool = True
    soft_stop_duration: float = 3.0
