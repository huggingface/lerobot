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

"""Startup joint-mismatch guard for teleoperation and policy control loops.

The very first action sent to a robot is unconditional: nothing verifies that the
teleoperator (or policy) and the follower agree on where the joints currently are.
Any constant offset between the two conventions -- a flipped sign, a stale homing
offset, a multi-turn encoder that re-woke on a different 2*pi branch after a power
cycle -- therefore materializes as a full-speed jump on frame zero, usually silently
clipped against the soft joint limits. This class turns that jump into either a
smooth ramp or a hard, explainable error.

The guard is robot-agnostic: it only inspects action/observation keys that end in
``.pos`` and are present in both dictionaries with float values. Everything else
(velocities, cameras, robots whose teleop does not speak joint space) passes
through untouched.

Typical wiring, once per control loop before ``robot.send_action``::

    guard = StartupJointGuard(threshold=10.0, ramp_duration_s=1.5)
    ...
    action = guard.process(action, obs, now=time.perf_counter())
    robot.send_action(action)
"""

import logging
import time

logger = logging.getLogger(__name__)

_POS_SUFFIX = ".pos"


class StartupJointGuard:
    """Detects and defuses first-frame joint mismatches between commanded and measured positions.

    On the first processed frame, every joint key ``{name}.pos`` present in both the
    action and the observation is compared. If all deltas are within ``threshold``
    the guard disarms immediately and becomes a pass-through. Otherwise:

    - ``mode="ramp"`` (default): the commanded positions are blended from the
      measured positions to the live targets over ``ramp_duration_s`` seconds
      (``cmd = measured_0 + alpha * (target - measured_0)``), so the robot converges
      to the leader/policy smoothly instead of jumping.
    - ``mode="abort"``: raises :class:`StartupMismatchError` listing the offending
      joints, so a badly calibrated convention is surfaced instead of executed.

    Args:
        threshold: Maximum tolerated per-joint |action - observation| on the first
            frame, in the robot's action units (degrees for followers that speak
            degrees, normalized units for normalized followers).
        ramp_duration_s: Duration of the blend-in ramp when a mismatch is detected.
        mode: ``"ramp"`` or ``"abort"``.
        enabled: If False the guard is a pass-through (single CLI switch-off).
    """

    def __init__(
        self,
        threshold: float = 10.0,
        ramp_duration_s: float = 1.5,
        mode: str = "ramp",
        enabled: bool = True,
    ):
        if mode not in ("ramp", "abort"):
            raise ValueError(f"Unsupported startup guard mode '{mode}'. Use 'ramp' or 'abort'.")
        if not (threshold > 0.0):
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if not (ramp_duration_s > 0.0):
            raise ValueError(f"ramp_duration_s must be > 0, got {ramp_duration_s}")
        self.threshold = threshold
        self.ramp_duration_s = ramp_duration_s
        self.mode = mode
        self.enabled = enabled

        self._armed = True  # True until the first frame has been inspected
        self._ramping = False
        self._ramp_start_ts: float | None = None
        self._ramp_base: dict[str, float] = {}  # measured positions captured on frame 0

    @property
    def is_ramping(self) -> bool:
        """Whether the guard is currently blending commands toward the live target."""
        return self._ramping

    @staticmethod
    def _joint_pairs(action: dict, observation: dict) -> dict[str, tuple[float, float]]:
        """Return ``{key: (commanded, measured)}`` for float ``.pos`` keys present in both dicts."""
        pairs = {}
        for key, cmd in action.items():
            if not key.endswith(_POS_SUFFIX):
                continue
            meas = observation.get(key)
            if isinstance(cmd, (int, float)) and isinstance(meas, (int, float)):
                pairs[key] = (float(cmd), float(meas))
        return pairs

    def process(self, action: dict, observation: dict, now: float | None = None) -> dict:
        """Inspect (and possibly rewrite) the action about to be sent to the robot.

        Args:
            action: The action dictionary bound for ``robot.send_action``.
            observation: The most recent ``robot.get_observation()`` result.
            now: Injectable monotonic timestamp [s] (defaults to ``time.perf_counter()``).

        Returns:
            The action to actually send: the input unchanged once disarmed, or a
            blended copy while ramping.

        Raises:
            StartupMismatchError: In ``"abort"`` mode when the first frame disagrees.
        """
        if not self.enabled:
            return action
        if now is None:
            now = time.perf_counter()

        if self._armed:
            self._armed = False
            pairs = self._joint_pairs(action, observation)
            offending = {
                key: cmd - meas for key, (cmd, meas) in pairs.items() if abs(cmd - meas) > self.threshold
            }
            if not offending:
                return action

            details = ", ".join(
                f"{key}: commanded {pairs[key][0]:.2f} vs measured {pairs[key][1]:.2f} (delta {delta:+.2f})"
                for key, delta in sorted(offending.items())
            )
            message = (
                f"Startup mismatch between commanded and measured joint positions "
                f"(threshold {self.threshold:.2f}): {details}. This usually means the "
                f"teleoperator and follower disagree on calibration conventions "
                f"(sign flip, stale zero, or a multi-turn encoder that wrapped after "
                f"a power cycle)."
            )
            if self.mode == "abort":
                raise StartupMismatchError(message)

            logger.warning("%s Ramping to the target over %.1f s.", message, self.ramp_duration_s)
            self._ramping = True
            self._ramp_start_ts = now
            self._ramp_base = {key: meas for key, (_cmd, meas) in pairs.items()}

        if self._ramping and self._ramp_start_ts is not None:
            alpha = (now - self._ramp_start_ts) / self.ramp_duration_s
            if alpha >= 1.0:
                self._ramping = False
                self._ramp_base = {}
                return action
            blended = dict(action)
            for key, base in self._ramp_base.items():
                cmd = blended.get(key)
                if isinstance(cmd, (int, float)):
                    blended[key] = base + alpha * (float(cmd) - base)
            return blended

        return action

    def reset(self) -> None:
        """Re-arm the guard (e.g. after a reconnection or a new episode)."""
        self._armed = True
        self._ramping = False
        self._ramp_start_ts = None
        self._ramp_base = {}


class StartupMismatchError(RuntimeError):
    """Raised in ``"abort"`` mode when the first commanded frame disagrees with the measured state."""
