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

"""Input conditioning and joint integration for the accessible teleoperator.

This module is deliberately free of I/O and of third-party dependencies so that the
behaviour an operator feels through the robot can be unit tested without a browser,
a camera, or a motor bus.

The pipeline for one joint is::

    raw channel value
      -> per-channel calibration (neutral, asymmetric comfortable range) -> [-1, 1]
      -> dead zone + gain curve                                          -> [-1, 1]
      -> exponential moving average                                      -> [-1, 1]
      -> multiply by the joint's maximum speed                           -> units/s
      -> integrate over a clamped time step and clamp to joint limits    -> units
"""

from dataclasses import dataclass, field

from .config_accessible_teleop import (
    ChannelCalibration,
    InputSource,
    JointBinding,
)

# Face channels rise and fall slowly compared with the control loop, so a small floor on
# the calibrated range keeps a nearly motionless operator from producing full-scale output.
MIN_CALIBRATED_RANGE = 1e-3

# Above the dead zone the response is slightly expansive, which buys fine resolution near
# neutral without giving up the top of the range.
RESPONSE_EXPONENT = 1.35

# Below this the smoothed axis is snapped to zero, so releasing an input actually stops the
# joint instead of leaving it creeping on the tail of the exponential decay.
AXIS_EPSILON = 1e-3


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def normalize_channel(value: float, calibration: ChannelCalibration) -> float:
    """Map a raw channel reading onto [-1, 1] using its calibrated comfortable range.

    The negative and positive sides are scaled independently. Most people cannot move a
    facial feature as far in one direction as the other, and forcing a symmetric range
    would either waste the larger side or make the smaller side unreachable.
    """
    delta = value - calibration.neutral
    span = calibration.negative_range if delta < 0 else calibration.positive_range
    return clamp(delta / max(span, MIN_CALIBRATED_RANGE), -1.0, 1.0)


def shape_axis(value: float, deadzone: float, gain: float) -> float:
    """Apply the dead zone and gain curve to a normalized axis."""
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    normalized = clamp((magnitude - deadzone) / max(1.0 - deadzone, MIN_CALIBRATED_RANGE), 0.0, 1.0)
    shaped = clamp(normalized**RESPONSE_EXPONENT * gain, 0.0, 1.0)
    return shaped if value > 0 else -shaped


@dataclass
class InputFrame:
    """One sample of everything the operator's input surface is reporting.

    Attributes:
        channels: Raw analog readings keyed by channel id, e.g. ``"face.mouth_open"`` or
            ``"pad_a.y"``. Face channels are uncalibrated; joystick axes are already in
            [-1, 1].
        keys: Currently held keys, keyed by browser ``KeyboardEvent.code``.
        engaged: Whether the operator has the clutch engaged. A frame that is not engaged
            produces no motion.
        tracking: Whether the camera is currently tracking a face. When this is False every
            face-driven joint stops, but keyboard and joystick joints keep working.
    """

    channels: dict[str, float] = field(default_factory=dict)
    keys: dict[str, bool] = field(default_factory=dict)
    engaged: bool = False
    tracking: bool = False


class JointController:
    """Turns operator input frames into absolute joint targets.

    The controller owns the commanded pose. It integrates velocity rather than mapping input
    position directly onto joint angle, because the operator's usable range of motion is far
    smaller than the robot's and a direct map would make most of the workspace unreachable.
    """

    def __init__(
        self,
        joints: list[str],
        bindings: dict[str, JointBinding],
        joint_limits: dict[str, tuple[float, float]],
        start_pose: dict[str, float],
        calibrations: dict[str, ChannelCalibration] | None = None,
        max_step_s: float = 0.06,
    ):
        self.joints = list(joints)
        self.bindings = bindings
        self.joint_limits = joint_limits
        self.start_pose = start_pose
        self.calibrations = calibrations if calibrations is not None else {}
        self.max_step_s = max_step_s

        self._pose: dict[str, float] = {}
        self._smoothed: dict[str, float] = {}
        self.reset()

    @property
    def pose(self) -> dict[str, float]:
        return dict(self._pose)

    @property
    def velocity(self) -> dict[str, float]:
        """Last commanded velocity per joint, in joint units per second."""
        return {
            joint: self._smoothed[joint] * self.bindings[joint].speed if joint in self.bindings else 0.0
            for joint in self.joints
        }

    def reset(self, pose: dict[str, float] | None = None) -> None:
        """Re-anchor the commanded pose and clear the smoothing state.

        Args:
            pose: Measured joint positions to anchor on, taken as given. A robot parked
                outside the configured limits is a fact, not an error, and clamping it here
                would turn re-anchoring into a movement command. Missing joints, and the
                configured start pose, are clamped because those are values a human typed.
        """
        if pose is None:
            self._pose = {
                joint: self._clamp_to_limits(joint, self.start_pose.get(joint, 0.0)) for joint in self.joints
            }
        else:
            self._pose = {
                joint: pose[joint]
                if joint in pose
                else self._clamp_to_limits(joint, self.start_pose.get(joint, 0.0))
                for joint in self.joints
            }
        self._smoothed = dict.fromkeys(self.joints, 0.0)

    def axis_for_joint(self, joint: str, frame: InputFrame) -> float:
        """Resolve a joint's binding against an input frame, returning a value in [-1, 1]."""
        binding = self.bindings.get(joint)
        if binding is None or binding.source is InputSource.NONE:
            return 0.0

        if binding.source is InputSource.KEYBOARD:
            # Keys are all-or-nothing, so the dead zone and gain curve have nothing to act on.
            value = 0.0
            if binding.positive_key and frame.keys.get(binding.positive_key):
                value += 1.0
            if binding.negative_key and frame.keys.get(binding.negative_key):
                value -= 1.0
            value = clamp(value, -1.0, 1.0)
        elif binding.source is InputSource.JOYSTICK:
            if not binding.channel:
                return 0.0
            value = shape_axis(
                clamp(frame.channels.get(binding.channel, 0.0), -1.0, 1.0),
                binding.deadzone,
                binding.gain,
            )
        elif binding.source is InputSource.FACE:
            if not binding.channel:
                return 0.0
            # An uncalibrated channel has no meaningful neutral, so treating its raw value as
            # a command would move the robot the instant a face appears.
            calibration = self.calibrations.get(binding.channel)
            if not frame.tracking or calibration is None or not calibration.ready:
                return 0.0
            value = shape_axis(
                normalize_channel(frame.channels.get(binding.channel, 0.0), calibration),
                binding.deadzone,
                binding.gain,
            )
        else:
            return 0.0

        return -value if binding.invert else value

    def step(self, frame: InputFrame, dt_s: float) -> dict[str, float]:
        """Advance the commanded pose by one control step.

        Args:
            frame: The latest input sample.
            dt_s: Elapsed time since the previous step. Clamped to ``max_step_s`` so that a
                stalled loop, a backgrounded browser tab, or a debugger breakpoint cannot
                turn into one large jump.

        Returns:
            The commanded joint positions, in the robot's normalized units.
        """
        dt_s = clamp(dt_s, 0.0, self.max_step_s)

        for joint in self.joints:
            binding = self.bindings.get(joint)
            if binding is None:
                continue

            target = self.axis_for_joint(joint, frame) if frame.engaged else 0.0
            smoothing = clamp(binding.smoothing, 0.0, 0.99)
            smoothed = self._smoothed[joint] * smoothing + target * (1.0 - smoothing)
            if abs(smoothed) < AXIS_EPSILON:
                smoothed = 0.0
            self._smoothed[joint] = smoothed

            if smoothed:
                current = self._pose[joint]
                self._pose[joint] = self._limit_travel(
                    joint, current, current + smoothed * binding.speed * dt_s
                )

        return self.pose

    def _clamp_to_limits(self, joint: str, value: float) -> float:
        low, high = self.joint_limits.get(joint, (-float("inf"), float("inf")))
        return clamp(value, low, high)

    def _limit_travel(self, joint: str, current: float, target: float) -> float:
        """Stop a joint leaving its limits without ever forcing it back inside them.

        A joint that starts beyond a limit, because the arm was parked at a mechanical stop
        or the limits were tightened between sessions, can still be driven back into range;
        it just cannot travel further out. Clamping outright would command a jump the
        operator never asked for.
        """
        low, high = self.joint_limits.get(joint, (-float("inf"), float("inf")))
        return clamp(target, min(low, current), max(high, current))
