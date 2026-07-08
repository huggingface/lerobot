#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Configuration for Quest VR arm teleoperator (via QuestArmTeleop)."""

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("quest_arm")
@dataclass
class QuestArmTeleopConfig(TeleoperatorConfig):
    """Configuration for Meta Quest VR arm teleoperator.

    Uses QuestArmTeleop's OculusReader (ADB) + ArmIK (Pinocchio/CasADi)
    to map Quest controller poses to NERO joint angles via IK.

    Attributes:
        ip_address: Quest IP for WiFi connection. None = USB.
        urdf_path: Absolute path to the NERO URDF file (required).
        package_dirs: Pinocchio mesh search directory (single path string).
        locked_joints: Joints excluded from IK (e.g. gripper joints).
        ee_parent_joint: Parent joint for the end-effector frame.
        ee_frame_name: Name of the EE frame added to the Pinocchio model.
        tool_pre_rot_rpy: Rotation from joint7 to TCP (RPY, radians).
        tool_translation_xyz: Translation from joint7 to TCP (meters).
        ros_to_arm_rpy: VR-to-arm frame alignment rotation (RPY, radians).
        ros_to_arm_xyz: VR-to-arm frame alignment translation (meters).
        w_pos: IK position error weight.
        w_ori: IK orientation error weight.
        w_reg: IK regularization weight (neutral pose).
        w_smooth: IK smoothness weight (temporal continuity).
        ipopt_max_iter: IPOPT maximum iterations.
        ipopt_tol: IPOPT convergence tolerance.
        start_button: Button to start teleoperation (A/B/X/Y).
        stop_button: Button to stop teleoperation (A/B/X/Y).
        trigger_axis: Button key for gripper analog trigger.
        hand_name: Which hand to use (right/left).
        gripper_max: Maximum gripper value (NERO: 0-100).
        gripper_min: Minimum gripper value.
        quest_arm_teleop_path: Path to QuestArmTeleop/src/oculus_reader/scripts.
            If None, auto-detected from standard location.
    """

    # VR connection
    ip_address: str | None = None

    # URDF / IK
    urdf_path: str | None = None
    package_dirs: str | None = None
    locked_joints: list[str] = field(
        default_factory=lambda: ["gripper_base_joint", "gripper_joint1", "gripper_joint2"]
    )
    ee_parent_joint: str = "joint7"
    ee_frame_name: str = "ee"
    tool_pre_rot_rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tool_translation_xyz: list[float] = field(default_factory=lambda: [0.1755, 0.0, -0.0235])

    # Coordinate alignment
    ros_to_arm_rpy: list[float] = field(default_factory=lambda: [-1.5708, 0.0, 0.0])
    ros_to_arm_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # IK weights
    w_pos: float = 2.0
    w_ori: float = 2.0
    w_reg: float = 0.01
    w_smooth: float = 0.1
    ipopt_max_iter: int = 50
    ipopt_tol: float = 1e-4

    # Button mapping
    start_button: str = "A"
    stop_button: str = "B"
    trigger_axis: str = "rightTrig"
    hand_name: str = "right"

    # Gripper
    gripper_max: float = 100.0
    gripper_min: float = 0.0

    # QuestArmTeleop source path
    quest_arm_teleop_path: str | None = None
