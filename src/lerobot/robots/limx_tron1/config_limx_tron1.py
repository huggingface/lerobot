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

"""Robot configuration for the LimX TRON1 legged robots.

TRON1 ships as several legged builds selected by the SDK ``ROBOT_TYPE`` variable (``PF_*``,
``SF_*``, ``WF_*``).  The build picks the joint layout -- a 6-joint point foot, or an 8-joint
sole/wheel foot -- so the joint table and gain defaults are derived from `robot_type` rather than
fixed.  Registering the subclass is what makes ``--robot.type=limx_tron1`` resolvable on the
LeRobot command line.
"""

from dataclasses import dataclass, field

from ..config import RobotConfig
from .joints import (
    VARIANT_DIMS,
    VARIANT_KD,
    VARIANT_KP,
    resolve_joint_names,
    resolve_joint_vector,
    variant_from_robot_type,
)


@RobotConfig.register_subclass("limx_tron1")
@dataclass
class LimxTron1Config(RobotConfig):
    """Connection and control settings for a LimX TRON1 legged robot.

    The robot is reached through the vendor low-level SDK, which opens a live connection to the
    robot's own controller.  The SDK addresses the actuated joints positionally, so the name
    table below is descriptive rather than SDK-provided; see
    [`limx_tron1.joints`][lerobot.robots.limx_tron1.joints] for where it comes from.

    Args:
        robot_ip (`str`, *optional*, defaults to `"10.192.1.2"`):
            Address of the robot's own control process.  The default is the address the
            vendor's SDK header documents for a real robot; your deployment may differ.
        robot_type (`str`, *optional*):
            Value to export as the SDK's ``ROBOT_TYPE`` variable before the vendor package is
            imported.  The expected value is the build, e.g. ``"PF_TRON1A"``, ``"SF_TRON1A"`` or
            ``"WF_TRON1A"``; only the ``PF_``/``SF_``/``WF_`` prefix matters here, as it selects
            the joint layout.  `None` selects the point-foot layout (6 joints).
        kp (`list[float]`, *optional*, defaults to `[]`):
            Position gains, one per joint in `joint_names` order.  An empty list selects the
            vendor's deployment values for the selected build; see the note below.
        kd (`list[float]`, *optional*, defaults to `[]`):
            Derivative gains, matching `kp` joint for joint.  An empty list selects the vendor's
            deployment values for the selected build.
        joint_names (`list[str]`, *optional*, defaults to `[]`):
            Joint name table in SDK vector order.  An empty list selects the vendor names for the
            selected build; override to match an existing dataset or policy checkpoint.
        publish_rate (`float`, *optional*, defaults to 500.0):
            Rate in Hz at which joint targets are republished to the robot while it is
            connected.  500 Hz matches the vendor's own TRON1 controller loop.
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
            Identifier that distinguishes this robot from other LimX TRON1 instances.
        calibration_dir (`Path`, *optional*):
            Directory LeRobot uses for calibration files.  Unused for this robot; see the note
            below.

    Note:
        Calibration is a no-op on this robot.  The vendor controller performs its own zeroing and
        bring-up, so LeRobot never writes a calibration file and ``calibration_dir`` is ignored.

    Note:
        The gain defaults are the vendor's deployment values for the TRON1A build of each family
        (``tron1-rl-deploy-python``, ``controllers/model/{PF,SF,WF}_TRON1A/params.yaml``), not an
        estimate.  They are one ``stiffness``/``damping`` pair reused across the leg joints, with a
        lower damping on the fourth per-leg joint of the sole- and wheel-foot builds.
    """

    # --- connection -------------------------------------------------------
    robot_ip: str = "10.192.1.2"
    robot_type: str | None = None

    # --- control ----------------------------------------------------------
    #: An empty list selects the vendor default for the build derived from `robot_type`.
    kp: list[float] = field(default_factory=list)
    kd: list[float] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    publish_rate: float = 500.0

    # --- bring-up pose ----------------------------------------------------
    start_pose: list[float] | None = None

    # --- safety -----------------------------------------------------------
    observation_timeout: float = 1.0
    state_max_age: float | None = None

    @property
    def variant(self) -> str:
        """The build family selected by `robot_type` (see
        [`limx_tron1.joints`][lerobot.robots.limx_tron1.joints])."""
        return variant_from_robot_type(self.robot_type)

    def __post_init__(self):
        """Validate the joint tables and numeric ranges at construction time."""
        super().__post_init__()
        variant = self.variant
        dim = VARIANT_DIMS[variant]

        # Resolve the joint table and gains for the selected build.  An empty list falls back to
        # the vendor default for that build; an explicit value is validated against the width.
        self.joint_names = list(resolve_joint_names(self.joint_names or None, variant))
        self.kp = list(VARIANT_KP[variant]) if not self.kp else list(resolve_joint_vector(self.kp, dim, "kp"))
        self.kd = list(VARIANT_KD[variant]) if not self.kd else list(resolve_joint_vector(self.kd, dim, "kd"))

        if self.start_pose is not None:
            self.start_pose = list(resolve_joint_vector(self.start_pose, dim, "start_pose"))
        if self.publish_rate <= 0:
            raise ValueError(f"publish_rate must be positive, got {self.publish_rate}")
        if self.observation_timeout <= 0:
            raise ValueError(f"observation_timeout must be positive, got {self.observation_timeout}")
        if self.state_max_age is not None and self.state_max_age <= 0:
            raise ValueError(f"state_max_age must be positive or None, got {self.state_max_age}")
