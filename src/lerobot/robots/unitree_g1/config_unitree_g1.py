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


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
   # id: str = "unitree_g1"
    simulation_mode: bool = True
    kp_high = 40.0
    kd_high = 3.0
    kp_low = 80.0
    kd_low = 3.0
    kp_wrist = 40.0
    kd_wrist = 1.5
    all_motor_q = None
    arm_velocity_limit = 100.0
    control_dt = 1.0 / 250.0

    all_motor_q = None
    arm_velocity_limit = 100.0
    control_dt = 1.0 / 250.0

    gradual_start_time: float | None = None
    gradual_time: float | None = None

    freeze_body: bool = False
    gravity_compensation: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Socket communication configuration (REQUIRED)
    # This robot class ONLY uses sockets to communicate with a bridge on the Orin
    # Run 'python dds_to_socket.py' on the Orin first, then set this to the Orin's IP
    # Example: socket_host="192.168.123.164" (Orin's wlan0 IP)
    socket_host: str | None = None# = "172.18.129.215"
    socket_port: int | None = None

    # Locomotion control
    locomotion_control: bool = False
    #policy_path: str = "src/lerobot/robots/unitree_g1/assets/g1/locomotion/motion.pt"
    policy_path: str = "src/lerobot/robots/unitree_g1/assets/g1/locomotion/GR00T-WholeBodyControl-Walk.onnx"
    
    # Locomotion parameters (from g1.yaml)
    locomotion_control_dt: float = 0.02
    
    leg_joint2motor_idx: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    locomotion_kps: list = field(default_factory=lambda: [150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40])
    locomotion_kds: list = field(default_factory=lambda: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2])
    default_leg_angles: list = field(default_factory=lambda: [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0])

    arm_waist_joint2motor_idx: list = field(default_factory=lambda: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    locomotion_arm_waist_kps: list = field(default_factory=lambda: [250, 250, 250, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20, 20, 20])
    locomotion_arm_waist_kds: list = field(default_factory=lambda: [5, 5, 5, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
    locomotion_arm_waist_target: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    locomotion_action_scale: float = 0.25
    cmd_scale: list = field(default_factory=lambda: [2.0, 2.0, 0.25])
    
    # GR00T-specific scaling (different from regular locomotion!)
    groot_ang_vel_scale: float = 0.25  # GR00T uses 0.5, not 0.25
    groot_cmd_scale: list = field(default_factory=lambda: [2.0, 2.0, 0.25])  # yaw is 0.5 for GR00T
    num_locomotion_actions: int = 12
    num_locomotion_obs: int = 47
    max_cmd: list = field(default_factory=lambda: [0.8, 0.5, 1.57])
    locomotion_imu_type: str = "pelvis"  # "torso" or "pelvis"