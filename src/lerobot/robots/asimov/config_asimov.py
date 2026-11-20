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
from math import degrees, isfinite

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

# Fixed before connecting so dataset features can be built offline. Names/order from
# menlo-sdk 0.1.0, menlo/asimov/robots.py: ASIMOV_1_BIPED_JOINTS.
ASIMOV_JOINTS = (
    "L_Hip_Pitch",
    "L_Hip_Roll",
    "L_Hip_Yaw",
    "L_Knee",
    "L_Ankle_A",
    "L_Ankle_B",
    "R_Hip_Pitch",
    "R_Hip_Roll",
    "R_Hip_Yaw",
    "R_Knee",
    "R_Ankle_A",
    "R_Ankle_B",
    "L_Shoulder_Pitch",
    "L_Shoulder_Roll",
    "L_Shoulder_Yaw",
    "L_Elbow",
    "L_Wrist_Yaw",
    "R_Shoulder_Pitch",
    "R_Shoulder_Roll",
    "R_Shoulder_Yaw",
    "R_Elbow",
    "R_Wrist_Yaw",
    "Waist_Yaw",
    "Neck_Yaw",
    "Neck_Pitch",
)
# Joints an action may drive in joint mode. The legs are held (as measured, or in `leg_stance`).
UPPER_BODY_JOINTS = ASIMOV_JOINTS[12:]
RIGHT_ARM_JOINTS = ("R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Shoulder_Yaw", "R_Elbow", "R_Wrist_Yaw")

CONTROL_MODES = ("velocity", "joints")
JOINT_GAINS = ("edge", "menlo_amp")


@RobotConfig.register_subclass("asimov")
@dataclass
class AsimovConfig(RobotConfig):
    """Asimov 1 over the Menlo SDK's UDP transport.

    ``control_mode="velocity"`` drives the onboard walking policy, which keeps balancing.
    ``control_mode="joints"`` puts every joint under position control with the walking policy
    off: actions are absolute targets for ``controlled_joints`` and the legs are held. Use it
    only with the robot supported (gantry rope taut) or in a stance known to stand.
    """

    host: str
    control_mode: str = "velocity"
    command_port: int = 8850
    # Receive LAN telemetry on all local interfaces; the SDK filters by robot source IP.
    state_host: str = "0.0.0.0"  # nosec B104
    state_port: int = 8851
    version_port: int = 3000
    connect_timeout_s: float = 5.0
    # Reject old observations, independently of the SDK's lost-link detection.
    max_state_age_s: float = 0.5
    # Each command expires unless refreshed. The SDK sends zero when this expires.
    command_timeout_s: float = 0.25
    # Body-frame forward/left limits (m/s), and counter-clockwise yaw (deg/s).
    max_linear_velocity: float = 0.3
    max_lateral_velocity: float = 0.2
    max_angular_velocity: float = 30.0
    # Cameras use the standard LeRobot backends; UDP does not carry onboard video.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # --- control_mode="joints" ---
    # Upper-body joints the action drives (firmware names, radians). Others hold their pose.
    controlled_joints: list[str] = field(default_factory=lambda: list(RIGHT_ARM_JOINTS))
    # "edge": Asimov Edge's own gain table. "menlo_amp": Menlo's hardware-tuned table
    # (asimov-1-amp), the leg stiffness the free-standing stance was tested with.
    joint_gains: str = "edge"
    # Legs (hip pitch, knee, ankle pitch) in rad for the left leg, mirrored on the right;
    # ankle rolls are kept. None holds the legs where they are when control is taken.
    leg_stance: list[float] | None = None
    # Largest change of a joint target per action (rad); larger requests are clipped.
    max_joint_step: float = 0.1
    # Optional absolute limits per controlled joint, e.g. {"R_Elbow": [-1.5, 0.2]} (rad).
    joint_limits: dict[str, list[float]] = field(default_factory=dict)
    # Glide into the held pose on the first action, and back to it before handing over.
    take_over_s: float = 2.0
    release_s: float = 4.0
    # Asimov Edge damps 2 s after the last setpoint: re-send the target after this long.
    keepalive_s: float = 0.2
    # Tilt watchdog (IMU, every 50 ms): hand back to the balance controller beyond either.
    max_tilt_deg: float = 16.0
    max_extra_tilt_deg: float = 9.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.host.strip():
            raise ValueError("host must be the Asimov robot's address.")
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}.")
        for name in ("command_port", "state_port", "version_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be an integer between 1 and 65535.")
        for name in (
            "connect_timeout_s",
            "max_state_age_s",
            "command_timeout_s",
            "max_joint_step",
            "take_over_s",
            "release_s",
            "keepalive_s",
            "max_tilt_deg",
            "max_extra_tilt_deg",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        # Keep recorded actions within the firmware's caps, not just the SDK's limits.
        for name, cap in (
            ("max_linear_velocity", 0.4),
            ("max_lateral_velocity", 0.4),
            ("max_angular_velocity", degrees(0.8)),
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= cap:
                raise ValueError(f"{name} must be finite and between 0 and {cap}.")
        self._validate_joint_options()

    def _validate_joint_options(self) -> None:
        joints = list(self.controlled_joints)
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("controlled_joints must list distinct joints.")
        if not set(joints) <= set(UPPER_BODY_JOINTS):
            raise ValueError(f"controlled_joints must be upper-body joints: {UPPER_BODY_JOINTS}.")
        if self.joint_gains not in JOINT_GAINS:
            raise ValueError(f"joint_gains must be one of {JOINT_GAINS}.")
        if self.keepalive_s >= 1.0:
            raise ValueError("keepalive_s must stay well under Asimov Edge's 2 s trajectory timeout.")
        if self.max_tilt_deg > 45:
            raise ValueError("max_tilt_deg must be at most 45.")
        if self.leg_stance is not None and (
            len(self.leg_stance) != 3 or not all(isfinite(v) for v in self.leg_stance)
        ):
            raise ValueError("leg_stance must be three finite angles: hip pitch, knee, ankle pitch (rad).")
        for name, bounds in self.joint_limits.items():
            if name not in joints:
                raise ValueError(f"joint_limits names {name!r}, which is not in controlled_joints.")
            if len(bounds) != 2 or not all(isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"joint_limits[{name!r}] must be [low, high] with low < high.")
