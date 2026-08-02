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
from enum import StrEnum

from ..config import TeleoperatorConfig


class InputSource(StrEnum):
    """Where a joint's motion command comes from."""

    NONE = "none"
    KEYBOARD = "keyboard"
    JOYSTICK = "joystick"
    FACE = "face"


# Facial channels the browser page reports. Each is a signed difference between two
# MediaPipe blendshapes, so a relaxed face sits near zero and the two directions of a
# movement land on opposite signs.
FACE_CHANNELS: tuple[str, ...] = (
    "face.brow_vertical",
    "face.lip_lateral",
    "face.mouth_open",
    "face.mouth_shape",
    "face.eye_wink",
)

# On-screen joystick axes. Two pads give four independent axes.
JOYSTICK_CHANNELS: tuple[str, ...] = ("pad_a.x", "pad_a.y", "pad_b.x", "pad_b.y")

SO101_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def default_joints() -> list[str]:
    return list(SO101_JOINTS)


def default_joint_limits() -> dict[str, tuple[float, float]]:
    """Conservative travel limits in the follower's normalized units.

    The five arm joints are in degrees measured from the middle of their calibrated range,
    which is what ``MotorNormMode.DEGREES`` expects; the gripper is a percentage of its
    calibrated span.

    Every SO-101 has a slightly different usable range, because the range is recorded by
    driving each joint into its mechanical stop during calibration. Commanding past that
    recorded range presses the joint into the stop, so the defaults sit inside the narrowest
    range we have measured. Widen them per joint from your own calibration file, where the
    half-range in degrees is ``(range_max - range_min) / 2 * 360 / 4095``.
    """
    return {
        "shoulder_pan": (-75.0, 75.0),
        "shoulder_lift": (-75.0, 75.0),
        "elbow_flex": (-75.0, 75.0),
        "wrist_flex": (-75.0, 75.0),
        "wrist_roll": (-75.0, 75.0),
        "gripper": (0.0, 100.0),
    }


def default_start_pose() -> dict[str, float]:
    """Pose the teleoperator commands the moment the operator first engages the clutch."""
    return {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }


@dataclass
class ChannelCalibration:
    """The comfortable range one operator can reach on one input channel.

    Attributes:
        neutral: The channel's reading when the operator is relaxed.
        negative_range: How far below neutral the operator can comfortably reach.
        positive_range: How far above neutral the operator can comfortably reach.
        ready: Whether a range has actually been captured. Face channels produce no motion
            until this is True.
    """

    neutral: float = 0.0
    negative_range: float = 1.0
    positive_range: float = 1.0
    ready: bool = False


@dataclass
class JointBinding:
    """How one robot joint is driven, and how firmly it responds.

    Every joint is tuned separately. An operator whose eyebrows move further than their
    lips needs different gain on each, and a joint that carries the arm's weight usually
    wants a lower speed than the gripper.

    Attributes:
        source: Which kind of input drives this joint.
        channel: Channel id for face and joystick sources, e.g. ``"face.mouth_open"``.
        positive_key: ``KeyboardEvent.code`` that drives the joint positive, for keyboard
            sources, e.g. ``"KeyF"``.
        negative_key: ``KeyboardEvent.code`` that drives the joint negative.
        invert: Swap the two directions.
        gain: Multiplier applied above the dead zone. Above 1.0 the operator reaches full
            speed without exhausting their full range of motion.
        deadzone: Fraction of the calibrated range treated as no motion, which keeps tremor
            and involuntary movement from reaching the robot.
        smoothing: Weight of the previous command in the exponential moving average, from
            0.0 for no smoothing to just under 1.0 for very heavy smoothing.
        speed: Joint units per second at full deflection. Degrees per second for the arm
            joints, percent per second for the gripper.
    """

    source: InputSource = InputSource.NONE
    channel: str | None = None
    positive_key: str | None = None
    negative_key: str | None = None
    invert: bool = False
    gain: float = 1.7
    deadzone: float = 0.12
    smoothing: float = 0.72
    speed: float = 30.0


def default_bindings() -> dict[str, JointBinding]:
    """No joint moves until the operator binds it, in the page or in the config.

    Binding every joint by default would hand a new operator six live axes at once, and the
    whole point of this teleoperator is that the right mapping is personal.
    """
    return {joint: JointBinding() for joint in SO101_JOINTS}


@TeleoperatorConfig.register_subclass("accessible_teleop")
@dataclass
class AccessibleTeleopConfig(TeleoperatorConfig):
    """Configuration for the browser-driven accessible teleoperator.

    Attributes:
        joints: Robot joints this teleoperator commands, in the order shown in the page.
        bindings: Per-joint input binding and tuning.
        joint_limits: Inclusive travel limits per joint, in the follower's normalized units.
        start_pose: Pose commanded when the operator first engages the clutch.
        host: Interface the control page is served on. Leave on loopback unless you have
            arranged TLS: browsers only grant camera access to secure contexts, and
            ``localhost`` counts as one while a bare LAN address does not.
        web_port: TCP port for the control page and its WebSocket.
        open_browser: Open the control page automatically on connect.
        connect_timeout_s: How long ``connect`` waits for the page to attach.
        input_timeout_s: If no input frame arrives within this window the teleoperator stops
            commanding motion, so a closed tab or a dead Wi-Fi link cannot leave the robot
            travelling.
        max_step_s: Largest time step the integrator will honour, which bounds how far a
            joint can move in a single control cycle.
        face_tracking: Serve the face-tracking channels. Turning this off leaves the
            joysticks and keyboard, and avoids loading MediaPipe entirely.
        mediapipe_base_url: Where the page loads the MediaPipe tasks-vision bundle and its
            WASM runtime from. Point this at a local mirror to work offline.
        face_landmarker_url: Where the page loads the face landmarker model from.
    """

    joints: list[str] = field(default_factory=default_joints)
    bindings: dict[str, JointBinding] = field(default_factory=default_bindings)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=default_joint_limits)
    start_pose: dict[str, float] = field(default_factory=default_start_pose)

    host: str = "127.0.0.1"
    web_port: int = 8777
    open_browser: bool = True
    connect_timeout_s: float = 180.0
    input_timeout_s: float = 0.5
    max_step_s: float = 0.06

    face_tracking: bool = True
    mediapipe_base_url: str = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35"
    face_landmarker_url: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
