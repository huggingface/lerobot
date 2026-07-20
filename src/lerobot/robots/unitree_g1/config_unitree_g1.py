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

    # Synthetic zero-image cameras exposed as ``observation.images.{name}`` (H×W×3
    # black frames). Lets image-conditioned policies (e.g. pi0.5 / OpenHLM) run in
    # sim before real cameras are wired. Empty = disabled.
    empty_cameras: list[str] = field(default_factory=list)
    empty_camera_hw: tuple[int, int] = (224, 224)

    # Publish Dex3 hand commands (``rt/dex3/{left,right}/cmd``) driven by the OpenHLM
    # gripper scalars (``wb.7.pos`` left, ``wb.15.pos`` right). Lets the 43-DoF sim
    # (or a real Dex3-equipped G1) show grasping. The scalar in [0, 1] is remapped to
    # a curl amount (``hand_open_grip_value`` -> open) and scaled onto
    # ``hand_closed_pose`` (7 joints: thumb_0/1/2, middle_0/1, index_0/1). Flip signs
    # in ``hand_closed_pose`` if fingers curl the wrong way.
    publish_hands: bool = False
    hand_open_grip_value: float = 1.0
    hand_closed_grip_value: float = 0.0
    hand_closed_pose: list[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.9, 1.3, 1.3, 1.3, 1.3]
    )
    hand_kp: float = 1.5
    hand_kd: float = 0.1

    # Replay recorded camera frames from a LeRobot parquet episode as the camera
    # feed (e.g. OpenHLM-data episode). Maps a robot camera name to a parquet image
    # column; frames advance one per observation and loop. Lets a VLA see the real
    # task video in sim without live cameras. Empty map = disabled.
    replay_camera_parquet: str | None = None
    replay_camera_map: dict[str, str] = field(default_factory=dict)
    replay_camera_loop: bool = True

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Locomotion controller class name, e.g. "GrootLocomotionController",
    # "HolosomaLocomotionController", or "SonicWholeBodyController". None disables it.
    controller: str | None = None

    # On disconnect (e.g. Ctrl-C), seconds to hold the current pose while ramping joint
    # stiffness (kp) to zero — a soft, damped settle instead of an instant limp /
    # free-fall. 0 disables it (immediate zero-torque). Real robot only.
    graceful_stop_s: float = 1.5
