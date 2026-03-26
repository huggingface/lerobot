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

from lerobot.cameras import CameraConfig

from ....config import RobotConfig


@RobotConfig.register_subclass("panthera_arm")
@dataclass
class PantheraArmConfig(RobotConfig):
    """
    Configuration for Panthera arm robot wrapper.

    This wrapper expects `Panthera_lib` to be extracted next to this module from
    `hardware/high_torque_robotics/Panthera_lib.zip` in Vector-Wangel/XLeRobot.
    Install the Panthera runtime dependencies separately.
    If `config_path` is unset, this wrapper expects
    `robot_param/Follower.yaml` to be present next to this module.
    """

    config_path: str | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Cartesian delta action scaling
    xy_step_m: float = 0.001
    vertical_step_m: float = 0.001
    rotation_step_deg: float = 0.2

    # IK options
    ik_max_iter: int = 1000
    ik_eps: float = 1e-3
    ik_damping: float = 1e-2
    ik_adaptive_damping: bool = True
    ik_multi_init: bool = False

    # Joint command options
    joint_velocity: list[float] = field(default_factory=lambda: [0.5] * 6)
    max_torque: list[float] | None = None

    # High-rate joint-impedance mode.
    use_joint_impedance: bool = False
    impedance_control_hz: float = 800.0
    impedance_k_joint: list[float] = field(default_factory=lambda: [20.0, 25.0, 25.0, 20.0, 10.0, 5.0])
    impedance_b_joint: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.8, 0.4, 0.2])
    tau_limit: list[float] = field(default_factory=lambda: [10.0, 20.0, 20.0, 10.0, 5.0, 5.0])
    enable_coriolis_comp: bool = False
    friction_fc: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.05, 0.0, 0.0])
    friction_fv: list[float] = field(default_factory=lambda: [0.03, 0.03, 0.03, 0.03, 0.0, 0.0])
    friction_vel_threshold: float = 0.02
    dq_lpf_cutoff_hz: float = 40.0
    enforce_joint_limit_margin: bool = True
    joint_limit_margin_rad: float = 0.1
    impedance_fail_safe_stop: bool = True
    impedance_max_consecutive_errors: int = 5
    impedance_error_log_interval_s: float = 1.0

    # Optional startup sequence inspired by the manufacturer script.
    # Disabled by default to avoid unexpected autonomous moves on connect.
    run_startup_sequence: bool = False
    startup_home_pos_m: list[float] = field(default_factory=lambda: [0.24, 0.0, 0.15])
    startup_home_euler_rad: list[float] = field(default_factory=lambda: [0.0, 1.5707963267948966, 0.0])
    startup_lift_m: float = 0.1
    startup_movej_duration_s: float = 3.0
    startup_lift_duration_s: float = 2.0

    # Gripper control options
    gripper_step: float = 0.02
    gripper_velocity: float = 0.5
    gripper_max_torque: float = 0.5
    gripper_kp: float = 8.0
    gripper_kd: float = 0.5

    # Optional workspace clamping
    min_radius_m: float = 0.05
    max_radius_m: float = 0.8
    min_z_m: float = -0.1
    max_z_m: float = 0.8

    stop_on_disconnect: bool = True
