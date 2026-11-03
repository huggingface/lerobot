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

"""Configuration shared by single-arm and bimanual YAM followers."""

import math
from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7))
MOTOR_NAMES = (*JOINT_NAMES, "gripper")
YAM_FEATURE_NAMES = tuple(f"{name}.pos" for name in MOTOR_NAMES)
# I2RT yam/v1/yam.xml; radians, without widening the manufacturer's limits.
JOINT_LIMITS_RAD = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)
JOINT_LIMITS_DEG = tuple((math.degrees(lower), math.degrees(upper)) for lower, upper in JOINT_LIMITS_RAD)


@dataclass
class YamFollowerConfig:
    """Settings of one YAM follower arm, shared by the single-arm and bimanual robots."""

    port: str
    expected_adapter_serial: str | None = None
    initial_position_deg: list[float] | None = field(default_factory=lambda: [0.0] * 6)
    initial_tolerance_deg: float = 11.5
    kp: list[float] = field(default_factory=lambda: [80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
    kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
    gripper_kp: float = 20.0
    gripper_kd: float = 0.5
    gripper_force_limit_n: float = 50.0
    max_joint_speed_deg_s: float = 17.0
    max_gripper_speed: float = 12.0
    max_tracking_error_deg: float = 8.5
    gravity_compensation: bool = True
    gravity_factors: list[float] = field(default_factory=lambda: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    idle_mode: Literal["hold", "float"] = "hold"
    float_kd: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
    friction_compensation: bool = False
    coulomb_friction: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 0.06, 0.06, 0.06])
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    use_velocity_and_torque: bool = False
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.2
    command_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "gravity_factors", "float_kd", "coulomb_friction"):
            values = getattr(self, name)
            if len(values) != 6:
                raise ValueError(f"{name} must contain six values; got {len(values)}")
        for name in (
            "initial_tolerance_deg",
            "gripper_force_limit_n",
            "max_joint_speed_deg_s",
            "max_gripper_speed",
            "max_tracking_error_deg",
            "control_frequency",
            "feedback_timeout_s",
            "command_timeout_s",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        if self.idle_mode == "float" and not self.gravity_compensation:
            raise ValueError(
                "idle_mode='float' needs gravity_compensation. "
                "Enable it with --robot.gravity_compensation=true."
            )
        if self.initial_position_deg is not None:
            if len(self.initial_position_deg) != 6:
                raise ValueError(
                    f"initial_position_deg must contain six values or be null; "
                    f"got {len(self.initial_position_deg)}"
                )
            for name, value, (lower, upper) in zip(
                JOINT_NAMES, self.initial_position_deg, JOINT_LIMITS_DEG, strict=True
            ):
                if not lower <= value <= upper:
                    raise ValueError(
                        f"{name} initial position must be between {lower:.1f} and {upper:.1f} degrees; "
                        f"got {value}"
                    )


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerRobotConfig(RobotConfig, YamFollowerConfig):
    """Configuration for a single YAM follower.

    Args:
        port (`str`):
            Linux SocketCAN interface used by the arm, for example `can0`.
        expected_adapter_serial (`str | None`, *optional*):
            USB adapter serial used to detect swapped CAN interfaces.
        initial_position_deg (`list[float] | None`, *optional*):
            Pose in degrees the arm must be in before torque is enabled; a check, not a motion command.
            Zeros are the manufacturer's folded pose; `None` accepts any valid pose.
        initial_tolerance_deg (`float`, *optional*, defaults to 11.5):
            Allowed difference from `initial_position_deg` before torque is enabled.
        kp (`list[float]`, *optional*):
            MIT position gains for the six arm joints.
        kd (`list[float]`, *optional*):
            MIT damping gains for the six arm joints.
        gripper_kp (`float`, *optional*, defaults to 20.0):
            MIT position gain for the gripper, I2RT's value for the linear DM4310 gripper.
        gripper_kd (`float`, *optional*, defaults to 0.5):
            MIT damping gain for the gripper, I2RT's value for the linear DM4310 gripper.
        gripper_force_limit_n (`float`, *optional*, defaults to 50.0):
            Maximum finger force after the gripper is blocked on an object.
        max_joint_speed_deg_s (`float`, *optional*, defaults to 17.0):
            Maximum commanded arm-joint speed.
        max_gripper_speed (`float`, *optional*, defaults to 12.0):
            Maximum commanded gripper speed, in full strokes per second.
        max_tracking_error_deg (`float`, *optional*, defaults to 8.5):
            Maximum difference between the measured and commanded joint positions.
        gravity_compensation (`bool`, *optional*, defaults to `True`):
            Whether to add gravity feed-forward from the YAM v1 model, downloaded on first use.
        gravity_factors (`list[float]`, *optional*):
            Per-joint multipliers applied to gravity feed-forward. Re-validate them after adding cameras or a
            payload, which the model doesn't include.
        idle_mode (`Literal["hold", "float"]`, *optional*, defaults to `"hold"`):
            Behavior before the first action and after the command timeout: `hold` keeps the pose, `float`
            lets the arm be moved by hand.
        float_kd (`list[float]`, *optional*):
            Per-joint damping gains used in float mode, I2RT's values.
        friction_compensation (`bool`, *optional*, defaults to `False`):
            Whether float mode compensates estimated Coulomb friction.
        coulomb_friction (`list[float]`, *optional*):
            Per-joint Coulomb friction estimates used in float mode, I2RT's values.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras attached to this arm.
        use_degrees (`bool`, *optional*, defaults to `True`):
            Whether joints use degrees and the gripper 0 to 100. `False` gives radians and 0 to 1, as in I2RT
            and MolmoAct2 data.
        use_velocity_and_torque (`bool`, *optional*, defaults to `False`):
            Whether observations include motor velocity and torque.
        control_frequency (`float`, *optional*, defaults to 100.0):
            Background servo-loop frequency in hertz. 100 Hz leaves classic CAN bandwidth for feedback and
            commands.
        feedback_timeout_s (`float`, *optional*, defaults to 0.2):
            Maximum time to wait for fresh feedback.
        command_timeout_s (`float`, *optional*, defaults to 1.0):
            Time without actions before entering the configured idle mode.
        id (`str | None`, *optional*):
            Robot identifier used to name the saved gripper calibration.
        calibration_dir (`pathlib.Path | None`, *optional*):
            Directory containing the gripper calibration file.
    """

    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        YamFollowerConfig.__post_init__(self)
