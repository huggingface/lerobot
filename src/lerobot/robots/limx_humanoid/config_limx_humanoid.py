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

"""Robot configuration for the LimX humanoid robots (Oli and Luna).

Registering the subclass is what makes ``--robot.type=limx_humanoid`` resolvable on the
LeRobot command line.
"""

from dataclasses import dataclass, field

from ..config import RobotConfig
from .joints import HUMANOID_DIM, resolve_joint_names, resolve_joint_vector

#: Default position gains, one per joint in `joint_names` order.
#:
#: Taken from the vendor's standing controller for the Oli build (``humanoid-rl-deploy-python``,
#: ``controllers/HU_D04_01/stand_controller/joint_params.yaml``).  They are the gains the vendor
#: ships to hold a real Oli upright, not an estimate.  Other control modes (walk, mimic) use
#: different gains, so override ``kp``/``kd`` for those -- but this is a sound starting point.
_DEFAULT_KP: tuple[float, ...] = (
    # left leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll
    580.0,
    500.0,
    500.0,
    660.0,
    400.0,
    400.0,
    # right leg
    580.0,
    500.0,
    500.0,
    660.0,
    400.0,
    400.0,
    # waist: yaw, roll, pitch
    800.0,
    800.0,
    800.0,
    # head: yaw, pitch
    10.0,
    10.0,
    # left arm: shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll
    80.0,
    80.0,
    80.0,
    50.0,
    40.0,
    20.0,
    10.0,
    # right arm
    80.0,
    80.0,
    80.0,
    50.0,
    40.0,
    20.0,
    10.0,
)

#: Default derivative gains, matching `_DEFAULT_KP` joint for joint.
_DEFAULT_KD: tuple[float, ...] = (
    # left leg
    8.0,
    6.0,
    6.0,
    8.0,
    2.0,
    2.0,
    # right leg
    8.0,
    6.0,
    6.0,
    8.0,
    2.0,
    2.0,
    # waist
    8.0,
    8.0,
    8.0,
    # head
    1.0,
    1.0,
    # left arm
    3.0,
    3.0,
    2.0,
    3.0,
    2.0,
    1.0,
    1.0,
    # right arm
    3.0,
    3.0,
    2.0,
    3.0,
    2.0,
    1.0,
    1.0,
)


@RobotConfig.register_subclass("limx_humanoid")
@dataclass
class LimxHumanoidConfig(RobotConfig):
    """Connection and control settings for a LimX humanoid robot.

    The robot is reached through the vendor low-level SDK, which opens a live connection to the
    robot's own controller.  The SDK addresses the 31 actuated joints positionally, so the name
    table below is descriptive rather than SDK-provided; see
    [`limx_humanoid.joints`][lerobot.robots.limx_humanoid.joints] for where it comes from.

    Args:
        robot_ip (`str`, *optional*, defaults to `"10.192.1.2"`):
            Address of the robot's own control process.  The default is the address the
            vendor's SDK header uses for a real humanoid; your deployment may differ.
        robot_type (`str`, *optional*):
            Value to export as the SDK's ``ROBOT_TYPE`` variable before the vendor package is
            imported.  The expected value is the robot build, e.g. ``"HU_D04_01"``.  Leave it as
            `None` when the SDK's own default is correct.  Note that on the humanoid build the SDK
            does not validate this value, so a wrong one is not reported as an error.
        kp (`list[float]`, *optional*):
            Position gains, one per joint in `joint_names` order.  Defaults to a conservative
            starting set; see the note below.
        kd (`list[float]`, *optional*):
            Derivative gains, matching `kp` joint for joint.
        joint_names (`list[str]`, *optional*):
            Joint name table in SDK vector order.  Defaults to the descriptive names in
            [`limx_humanoid.joints`][lerobot.robots.limx_humanoid.joints]; override to match an
            existing dataset or policy checkpoint.
        publish_rate (`float`, *optional*, defaults to 1000.0):
            Rate in Hz at which joint targets are republished to the robot while it is
            connected.  1000 Hz matches the vendor's own standing controller.
        start_pose (`list[float]`, *optional*):
            Joint positions, in `joint_names` order, to hold once the connection is established.
            `None` leaves the robot holding whatever pose it was in.
        observation_timeout (`float`, *optional*, defaults to 1.0):
            How long to wait for a fresh state sample before giving up.
        state_max_age (`float`, *optional*):
            Reject a state sample older than this many seconds.  `None` disables the check, which
            is only appropriate for transports that do not timestamp state; on real hardware
            prefer ~0.5.
        id (`str`, *optional*):
            Identifier that distinguishes this robot from other LimX humanoid instances.
        calibration_dir (`Path`, *optional*):
            Directory LeRobot uses for calibration files.  Unused for this robot; see the note
            below.

    Note:
        Calibration is a no-op on this robot.  The vendor controller performs its own zeroing and
        bring-up, so LeRobot never writes a calibration file and ``calibration_dir`` is ignored.

    Note:
        The gain defaults are a starting point, not vendor-published values.  Confirm them on the
        real robot before running a trained policy: gains that are too low will let a full-size
        humanoid sag under its own weight, and gains that are too high will make it oscillate.
    """

    # --- connection -------------------------------------------------------
    robot_ip: str = "10.192.1.2"
    robot_type: str | None = None

    # --- control ----------------------------------------------------------
    kp: list[float] = field(default_factory=lambda: list(_DEFAULT_KP))
    kd: list[float] = field(default_factory=lambda: list(_DEFAULT_KD))
    joint_names: list[str] = field(default_factory=lambda: list(resolve_joint_names(None)))
    publish_rate: float = 1000.0

    # --- bring-up pose ----------------------------------------------------
    start_pose: list[float] | None = None

    # --- safety -----------------------------------------------------------
    observation_timeout: float = 1.0
    state_max_age: float | None = None

    def __post_init__(self):
        """Validate the joint tables and the numeric ranges at construction time."""
        super().__post_init__()
        # Validate eagerly so a bad table fails at CLI parse time rather than halfway
        # through connect().
        self.joint_names = list(resolve_joint_names(self.joint_names))
        self.kp = list(resolve_joint_vector(self.kp, "kp"))
        self.kd = list(resolve_joint_vector(self.kd, "kd"))

        if self.start_pose is not None:
            self.start_pose = list(resolve_joint_vector(self.start_pose, "start_pose"))
        if self.publish_rate <= 0:
            raise ValueError(f"publish_rate must be positive, got {self.publish_rate}")
        if self.observation_timeout <= 0:
            raise ValueError(f"observation_timeout must be positive, got {self.observation_timeout}")
        if self.state_max_age is not None and self.state_max_age <= 0:
            raise ValueError(f"state_max_age must be positive or None, got {self.state_max_age}")
        if len(self.kp) != HUMANOID_DIM or len(self.kd) != HUMANOID_DIM:
            raise ValueError("kp and kd must each hold one value per joint")
