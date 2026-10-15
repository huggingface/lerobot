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

"""Robot configuration for the LimX Dynamics Luna humanoid robot (``HU_L04``).

Luna is a separate build from the Oli humanoid (``HU_D04``, robot type ``limx_humanoid``): it has
27 actuated joints rather than 31, so it gets its own robot type and joint table.

Registering the subclass is what makes ``--robot.type=limx_luna`` resolvable on the
LeRobot command line.
"""

from dataclasses import dataclass, field

from ..config import RobotConfig
from .joints import LUNA_DIM, resolve_joint_names, resolve_joint_vector


@RobotConfig.register_subclass("limx_luna")
@dataclass
class LimxLunaConfig(RobotConfig):
    """Connection and control settings for a LimX Luna robot.

    The robot is reached through the vendor low-level SDK, which opens a live connection to the
    robot's own controller.  The SDK addresses the 27 actuated joints positionally, so the name
    table below is descriptive rather than SDK-provided; see
    [`limx_luna.joints`][lerobot.robots.limx_luna.joints] for where it comes from.

    Args:
        robot_ip (`str`, *optional*, defaults to `"10.192.1.2"`):
            Address of the robot's own control process.  The default is the address the
            vendor's SDK header uses for a real humanoid; your deployment may differ.
        robot_type (`str`, *optional*):
            Value to export as the SDK's ``ROBOT_TYPE`` variable before the vendor package is
            imported.  For Luna this is ``"HU_L04_01"``.  Leave it as `None` only when the SDK's
            own default is already correct.  Note that on the humanoid build the SDK does not
            validate this value, so a wrong one is not reported as an error.
        kp (`list[float]`, *optional*):
            Position gains, one per joint in `joint_names` order.  Must be set: there is no
            default, and leaving it `None` is rejected at construction; see the note below.
        kd (`list[float]`, *optional*):
            Derivative gains, matching `kp` joint for joint.  Must be set, like `kp`.
        joint_names (`list[str]`, *optional*):
            Joint name table in SDK vector order.  Defaults to the vendor names in
            [`limx_luna.joints`][lerobot.robots.limx_luna.joints]; override to match an
            existing dataset or policy checkpoint.
        publish_rate (`float`, *optional*, defaults to 1000.0):
            Rate in Hz at which joint targets are republished to the robot while it is
            connected.  1000 Hz matches the loop rate of the vendor's own Luna controller.
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
            Identifier that distinguishes this robot from other LimX Luna instances.
        calibration_dir (`Path`, *optional*):
            Directory LeRobot uses for calibration files.  Unused for this robot; see the note
            below.

    Note:
        Calibration is a no-op on this robot.  The vendor controller performs its own zeroing and
        bring-up, so LeRobot never writes a calibration file and ``calibration_dir`` is ignored.

    Note:
        ``kp`` and ``kd`` have no default.  Unlike Oli, the vendor has not published standing
        gains for Luna.  The only Luna gains it has published are those of its motion-tracking
        policies (``luna-beyondmimic``), which are defined in motor space for the ankle and waist
        linkages and ordered differently, so they cannot be reused here as joint-space defaults.
        A policy should run with the same gains it was trained or recorded with; pass them
        explicitly, in `joint_names` order.
    """

    # --- connection -------------------------------------------------------
    robot_ip: str = "10.192.1.2"
    robot_type: str | None = None

    # --- control ----------------------------------------------------------
    kp: list[float] | None = None
    kd: list[float] | None = None
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
        if self.kp is None or self.kd is None:
            raise ValueError(
                "limx_luna has no default gains: pass kp and kd explicitly, one value per joint "
                f"in joint_names order ({LUNA_DIM} each), e.g. --robot.kp='[...]' --robot.kd='[...]'"
            )
        self.kp = list(resolve_joint_vector(self.kp, "kp"))
        self.kd = list(resolve_joint_vector(self.kd, "kd"))
        if any(gain < 0 for gain in (*self.kp, *self.kd)):
            raise ValueError("kp and kd must be non-negative")

        if self.start_pose is not None:
            self.start_pose = list(resolve_joint_vector(self.start_pose, "start_pose"))
        if self.publish_rate <= 0:
            raise ValueError(f"publish_rate must be positive, got {self.publish_rate}")
        if self.observation_timeout <= 0:
            raise ValueError(f"observation_timeout must be positive, got {self.observation_timeout}")
        if self.state_max_age is not None and self.state_max_age <= 0:
            raise ValueError(f"state_max_age must be positive or None, got {self.state_max_age}")
