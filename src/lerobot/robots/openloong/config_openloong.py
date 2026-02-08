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

"""
Configuration for OpenLoong "Qinglong" humanoid robot.

OpenLoong is a humanoid robot from Shanghai Humanoid Robotics Innovation Center,
using MPC (Model Predictive Control) and WBC (Whole-Body Control).

Reference: https://github.com/loongOpen/OpenLoong-Dyn-Control
"""

from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .openloong_utils import OPENLOONG_DEFAULT_GAINS, OPENLOONG_DEFAULT_STANDING_POSITION


def _build_gains() -> tuple[list[float], list[float]]:
    """Build kp and kd lists from body-part groupings."""
    kp = [v for g in OPENLOONG_DEFAULT_GAINS.values() for v in g["kp"]]
    kd = [v for g in OPENLOONG_DEFAULT_GAINS.values() for v in g["kd"]]
    return kp, kd


_DEFAULT_KP, _DEFAULT_KD = _build_gains()


@RobotConfig.register_subclass("openloong")
@dataclass
class OpenLoongConfig(RobotConfig):
    """
    Configuration for OpenLoong humanoid robot.
    
    OpenLoong uses MPC (Model Predictive Control) and WBC (Whole-Body Control)
    for dynamic locomotion including walking, jumping, and blind obstacle stepping.
    
    Args:
        kp: Position control gains for each joint (29 values)
        kd: Velocity control gains for each joint (29 values)
        default_positions: Default standing position for all 29 joints
        control_dt: Control loop timestep (default 1/500 = 500Hz for MPC)
        is_simulation: Whether to use MuJoCo simulation
        robot_ip: IP address for physical robot connection
        mpc_weights: MPC cost function weights
        wbc_task_priorities: WBC task priority ordering
        cameras: Camera configurations for vision
    """
    
    # PD control gains
    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # Default joint positions (standing pose)
    default_positions: list[float] = field(
        default_factory=lambda: OPENLOONG_DEFAULT_STANDING_POSITION.copy()
    )

    # Control loop timestep
    # OpenLoong uses 500Hz for MPC and 1kHz for WBC
    control_dt: float = 1.0 / 500.0  # 500Hz for MPC

    # Simulation mode
    is_simulation: bool = True

    # Robot network configuration
    robot_ip: str = "192.168.1.100"  # Default OpenLoong IP
    robot_port: int = 8888

    # MPC (Model Predictive Control) parameters
    mpc_horizon: float = 1.0  # Prediction horizon in seconds
    mpc_dt: float = 0.01  # MPC timestep
    mpc_u_weight: float = 0.001  # Input weight
    # State error weights: [eul_x, eul_y, eul_z, pos_x, pos_y, pos_z, omega_x, omega_y, omega_z, vel_x, vel_y, vel_z]
    mpc_L_diag: list[float] = field(
        default_factory=lambda: [100.0, 100.0, 100.0, 100.0, 100.0, 1000.0,
                                 10.0, 10.0, 10.0, 100.0, 100.0, 100.0]
    )
    # Input weights: [fl_x, fl_y, fl_z, tl_x, tl_y, tl_z, fr_x, fr_y, fr_z, tr_x, tr_y, tr_z]
    mpc_K_diag: list[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    # WBC (Whole-Body Control) parameters
    wbc_qp_weight_Q1: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )  # Contact force error weight
    wbc_qp_weight_Q2: list[float] = field(
        default_factory=lambda: [1.0] * 29
    )  # Joint acceleration error weight

    # Gait parameters
    gait_period: float = 0.4  # Walking gait period in seconds
    gait_duty_cycle: float = 0.5  # Duty cycle (stance phase ratio)
    step_height: float = 0.08  # Step height in meters

    # Safety limits
    max_joint_velocity: float = 10.0  # rad/s
    max_joint_torque: float = 100.0  # Nm
    max_base_linear_velocity: float = 1.0  # m/s
    max_base_angular_velocity: float = 1.0  # rad/s

    # Control mode: "position", "velocity", "torque", or "hybrid"
    control_mode: Literal["position", "velocity", "torque", "hybrid"] = "position"

    # Cameras (ZMQ-based remote cameras or local)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # MuJoCo model path (for simulation)
    mjcf_path: str | None = None  # Path to MJCF/URDF model

    # Debug options
    verbose: bool = False
    log_level: str = "INFO"
