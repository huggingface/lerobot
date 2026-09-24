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

"""Robot configuration for the LimX TRON2 dual-arm robot.

Registering the subclass is what makes ``--robot.type=limx_tron2`` resolvable
on the LeRobot command line.
"""

from dataclasses import dataclass

from ..config import RobotConfig
from .joints import resolve_gripper_names, resolve_joint_names


@RobotConfig.register_subclass("limx_tron2")
@dataclass
class LimxTron2Config(RobotConfig):
    """Connection and bring-up settings for a TRON2 dual-arm robot.

    ``init_joints`` / ``init_head`` are the poses the robot is commanded to hold
    while the connection is established.  Leaving them as ``None`` keeps the
    runtime's own defaults, which is the right choice unless a deployment has a
    specific bring-up pose.

    Args:
        robot_ip (`str`, *optional*, defaults to `"192.168.1.150"`):
            Address of the robot-side WebSocket controller.
        port (`int`, *optional*, defaults to 5000):
            TCP port that controller listens on.  Unlike the serial ``port`` of LeRobot's arm robots,
            this is a network port and is read together with ``robot_ip``.
        init_joints (`list[float]`, *optional*):
            Fourteen arm joint values, in ServoJ order, to hold while the connection is established.
        init_head (`list[float]`, *optional*):
            Two head joint values to hold while the connection is established.
        init_ee_z_min (`float`, *optional*, defaults to -0.6):
            Lowest end-effector height the bring-up pose is allowed to use.
        publish_rate (`float`, *optional*, defaults to 300.0):
            Rate in Hz at which the controller publishes state and accepts commands.
        eta_default (`float`, *optional*, defaults to `1.0 / 30.0`):
            Time constant the controller uses to interpolate towards a setpoint.
        state_max_age (`float`, *optional*):
            Reject a state sample older than this many seconds.  `None` disables the check, which is only
            appropriate for transports that do not timestamp state; on real hardware prefer ~0.5.
        observation_timeout (`float`, *optional*, defaults to 1.0):
            How long to wait for a fresh state sample before giving up.
        include_grippers (`bool`, *optional*, defaults to `False`):
            Expose the two grippers in the observation and action spaces.  Grippers are opt-in because the
            runtime normalises gripper *commands* to 0..1 but does not document the units of the value it
            reports back.
        joint_names (`list[str]`, *optional*):
            Joint name table in ServoJ order.  Defaults to the descriptive names in `joints`; override to
            match an existing dataset or policy checkpoint.
        gripper_names (`list[str]`, *optional*):
            Names for the left and right gripper.
        id (`str`, *optional*):
            Identifier that distinguishes this robot from other TRON2 instances.
        calibration_dir (`Path`, *optional*):
            Directory LeRobot uses for calibration files.  Unused for TRON2; see the note below.

    Note:
        Calibration is a no-op on this robot.  TRON2 performs its own zeroing and bring-up inside the robot
        controller, so LeRobot never writes a calibration file and ``calibration_dir`` is ignored.
    """

    # --- connection -------------------------------------------------------
    robot_ip: str = "192.168.1.150"
    port: int = 5000

    # --- bring-up pose ----------------------------------------------------
    init_joints: list[float] | None = None  # 14 values, arms only
    init_head: list[float] | None = None  # 2 values
    init_ee_z_min: float | None = -0.6

    # --- control loop -----------------------------------------------------
    publish_rate: float = 300.0
    eta_default: float = 1.0 / 30.0

    # --- safety -----------------------------------------------------------
    # Reject state older than this many seconds.  None disables the check,
    # which is only appropriate for transports that do not timestamp state
    # (e.g. the in-memory test transport).  On real hardware prefer ~0.5.
    state_max_age: float | None = None

    # How long to wait for a fresh state sample before giving up.
    observation_timeout: float = 1.0

    # --- action space -----------------------------------------------------
    # Grippers are opt-in.  The runtime normalises gripper *commands* to 0..1
    # but does not document the units of the value it reports back, so enabling
    # this before verifying that on hardware would put the observation and the
    # action on different scales.  See the "Grippers" section of the README.
    include_grippers: bool = False

    # --- naming -----------------------------------------------------------
    # Override to match an existing dataset or policy checkpoint.  Left as None
    # the descriptive defaults in .joints are used.
    joint_names: list[str] | None = None
    gripper_names: list[str] | None = None

    def __post_init__(self):
        """Validate the joint tables and the numeric ranges at construction time."""
        super().__post_init__()
        # Validate eagerly so a bad name table fails at CLI parse time rather
        # than halfway through connect().
        self.joint_names = list(resolve_joint_names(self.joint_names))
        self.gripper_names = list(resolve_gripper_names(self.gripper_names))

        if self.init_joints is not None and len(self.init_joints) != 14:
            raise ValueError(f"init_joints must have 14 values (both arms), got {len(self.init_joints)}")
        if self.init_head is not None and len(self.init_head) != 2:
            raise ValueError(f"init_head must have 2 values, got {len(self.init_head)}")
        if self.publish_rate <= 0:
            raise ValueError(f"publish_rate must be positive, got {self.publish_rate}")
        if self.observation_timeout <= 0:
            raise ValueError(f"observation_timeout must be positive, got {self.observation_timeout}")
        if self.state_max_age is not None and self.state_max_age <= 0:
            raise ValueError(f"state_max_age must be positive or None, got {self.state_max_age}")
