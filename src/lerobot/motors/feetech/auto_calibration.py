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

"""Automatic calibration of arms built from Feetech STS servos, such as the SO-101 leader and follower.

Instead of a person moving every joint through its range, the arm drives each joint slowly into its two
mechanical end stops at a limited torque, and the calibration (Homing_Offset, Min/Max_Position_Limit) is
computed from where the joints stalled. Which joints move together, and in which pose each end is measured,
is data: an :class:`AutoCalibrationPlan` (the SO arms' plan is in
``lerobot.robots.so_follower.auto_calibration_plan``).

Every move, the search for end stops included, is made in position mode, and the engine moves the joints itself: it
steps each joint's goal at the configured velocity and never lets it run more than a few degrees ahead of the joint.
A joint that meets a stop or an obstacle therefore pushes with a limited force, stops pushing as soon as it is held,
and a goal left behind by a dead process is a few degrees ahead at most. Velocity mode is not used, because on
STS3215 firmware 3.10 it reports Present_Position without Homing_Offset, and a servo switched back to position mode
while its torque is on can stay in velocity mode although Operating_Mode reads 0 (it then runs at whatever
Goal_Velocity says, ignoring Goal_Position, until it is switched to velocity mode and back with torque off).

The servos' own motion profile is not used either. A servo does not jump to a new Goal_Position: its setpoint
travels there at the speed Goal_Velocity had when it set off, and a speed or a goal written on the way does not
stop it. A joint blocked on the way to a distant goal goes on pushing at the torque limit until the setpoint
arrives, and after a new goal until the setpoint has come back (a wrist pushed into its stop for about 8 s that
way). So Goal_Velocity is 0 (no speed cap) throughout the run, and the setpoint follows the stepped goal at once.

What the engine guarantees:

- A goal equal to the present position is written, one acknowledged write per servo and read back, before torque is
  enabled, so no joint jumps; a joint that moves anyway when torque comes on ends the run.
- The servos are written with ``Lock=1`` during the run, so register changes stay in RAM and a power cut
  brings the previous calibration back. The caller writes the result to EEPROM.
- A timeout, a joint that cannot unfold, a range the plan considers implausible, an error or Ctrl-C ends the
  run without a result, and every register the run changed is written back. Torque is off at the end in
  every case: an arm that was not folded back by the plan drops.

Adapted from the automatic calibration by Isaac Sin (Maker-Mods) in huggingface/lerobot#3282.
"""

import logging
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from ..motors_bus import MotorCalibration

if TYPE_CHECKING:
    from .feetech import FeetechMotorsBus

logger = logging.getLogger(__name__)

FULL_TURN = 4096
# Present_Position of the middle of a range, as SerialMotorsBus.set_half_turn_homings() sets it.
HALF_TURN = 2047
# Homing_Offset is sign-magnitude with the sign in bit 11.
MAX_HOMING_OFFSET = 2047
STATUS_OVERLOAD = 0x20
POSITION_MODE = 0

# Registers the run changes. They are read before it starts and written back at the end; the calibration
# registers only when the run fails, since the caller writes the new calibration over them.
RESTORED_REGISTERS = ("Operating_Mode", "Torque_Limit", "Acceleration", "Goal_Velocity", "P_Coefficient")
CALIBRATION_REGISTERS = ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")

TARGETS = ("start", "unfolded", "fold", "unfold", "low", "high", "mid")
FIRST_ENDS = ("fold", "unfold", "positive", "negative")
CONTACT_CHECKS = ("warn", "abort", "off")


class AutoCalibrationError(RuntimeError):
    """The run stopped without a calibration. The servos' registers have been restored."""


def deg_to_steps(deg: float) -> int:
    return round(deg * FULL_TURN / 360)


def steps_to_deg(steps: float) -> float:
    return steps * 360 / FULL_TURN


def wrap(steps: int) -> int:
    """Shortest signed distance on the encoder circle, in [-2048, 2047]."""
    return (steps + FULL_TURN // 2) % FULL_TURN - FULL_TURN // 2


def homing_offset_for(raw: int, present: int = HALF_TURN) -> int:
    """Homing_Offset that makes the raw encoder position `raw` read `present` (Present = Actual - Offset)."""
    return max(-MAX_HOMING_OFFSET, min(MAX_HOMING_OFFSET, wrap(raw - present)))


@dataclass
class AutoCalibrationConfig:
    # Speed of the end-stop search and of every move, in encoder steps per second (4096 steps per turn).
    velocity: int = 200
    # Torque_Limit during the run, in 0.1 % of the servo's maximum torque. Never above the servo's
    # Max_Torque_Limit. None: the plan's torque limit for that arm.
    torque_limit: int | None = None
    # Acceleration register during the run.
    acceleration: int = 50
    # P_Coefficient of the position loop during the run. None keeps each servo's own value.
    p_coefficient: int | None = None
    # A move is done when the joint ends within this many degrees of its goal, or when it covered at least
    # `min_progress` of the way while its load stays below `blocked_load` times the torque limit: the
    # position loop then sags under the arm's weight, and nothing blocks it.
    move_tolerance_deg: float = 5.0
    min_progress: float = 0.5
    blocked_load: float = 0.9
    # Distance kept from a measured end stop when the plan moves a joint to that end.
    end_margin_deg: float = 3.0
    # A joint pushed towards an end stop, or on its way to a goal, rests against something when it lags its goal by
    # more than `stall_lag_deg` and moved less than `stall_progress_deg` in the last `stall_window_s`. The goal runs
    # at most three times the lag ahead of the joint, which bounds the push.
    stall_lag_deg: float = 4.0
    stall_progress_deg: float = 0.5
    stall_window_s: float = 0.3
    poll_interval_s: float = 0.05
    # Time after torque comes on before checking that no joint moved.
    start_delay_s: float = 0.5
    # A joint that holds still while another one reaches an end stop, and whose load then changes by at least
    # this much (in 0.1 % of full torque), shows an external contact: the arm pushes on the table or a clamp
    # rather than resting against the joint's own stop. "abort" ends the run, "warn" logs it, "off" skips it.
    contact_check: str = "abort"
    contact_load_delta: int = 150
    # Refuse ranges outside the span the plan gives for each joint.
    check_ranges: bool = True
    # Extra attempts for every bus transfer.
    num_retry: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.velocity <= 3000:
            raise ValueError(f"velocity must be in (0, 3000] steps/s, got {self.velocity}")
        if self.torque_limit is not None and not 0 < self.torque_limit <= 1000:
            raise ValueError(f"torque_limit must be in (0, 1000], got {self.torque_limit}")
        if self.contact_check not in CONTACT_CHECKS:
            raise ValueError(f"contact_check must be one of {CONTACT_CHECKS}, got {self.contact_check!r}")
        if not 0 < self.min_progress <= 1 or not 0 < self.blocked_load <= 1:
            raise ValueError("min_progress and blocked_load are fractions in (0, 1]")
        if not 0 < self.stall_progress_deg < self.stall_lag_deg:
            raise ValueError("stall_progress_deg must be positive and below stall_lag_deg")


# ---------------------------------------------------------------------------------------------------------
# The plan: what moves when
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JointRange:
    """The span, in degrees, that a joint's measured range must have to be accepted."""

    min_deg: float
    max_deg: float
    # A joint without end stops (it turned a full turn without stalling) gets the full encoder range with
    # its start pose in the middle, as wrist_roll does in the manual calibration.
    full_turn_ok: bool = False


@dataclass(frozen=True)
class Unfold:
    """Move one joint `deg` degrees away from the folded start pose, in its unfold direction.

    `sign` is that direction in Present_Position terms, +1 or -1. It is fixed by how the servo is mounted in the
    arm, so the plan states it rather than having the joint find out: a joint that starts some way from its fold
    stop moves freely towards it too, and would take the fold direction for the unfold one. A joint blocked on
    its way ends the run.
    """

    joint: str
    deg: float
    sign: int


@dataclass(frozen=True)
class Move:
    """Move several joints at once in position mode, and check that each one gets there.

    A target is "start" (where the joint was when the run began), "unfolded" (where its Unfold step took it),
    "fold" or "unfold" (the measured end in that direction, less the end margin), "low", "high" or "mid" of
    the measured range, or a number: degrees from the fold end towards the unfold end.
    """

    targets: Mapping[str, str | float]


@dataclass(frozen=True)
class Sweep:
    """Find both end stops of these joints, pushed together in position mode, while all others hold.

    `first` is the end each joint drives to first: "fold", "unfold", "positive" or "negative".
    """

    joints: tuple[str, ...]
    first: str = "positive"


Step = Unfold | Move | Sweep


@dataclass(frozen=True)
class AutoCalibrationPlan:
    """How an arm calibrates itself: the pose it starts in, the span each joint may have, and the steps."""

    start_pose: str
    joints: Mapping[str, JointRange]
    steps: tuple[Step, ...]
    # Torque_Limit of the run unless AutoCalibrationConfig.torque_limit is set, in 0.1 % of the servos' maximum
    # torque: enough for them to hold the stretched arm, and little more.
    torque_limit: int = 500

    def validate(self, motors: Iterable[str]) -> None:
        """Raise ValueError unless the steps measure every motor once and only use what they measured."""
        motors = set(motors)
        if set(self.joints) != motors:
            raise ValueError(f"The plan covers {sorted(self.joints)}, the bus has {sorted(motors)}.")
        unfolded: set[str] = set()
        measured: set[str] = set()

        def known(joint: str, step: Step) -> None:
            if joint not in motors:
                raise ValueError(f"{step}: unknown joint '{joint}'.")

        for step in self.steps:
            if isinstance(step, Unfold):
                known(step.joint, step)
                if step.sign not in (1, -1):
                    raise ValueError(f"{step}: sign must be +1 or -1.")
                if step.joint in measured:
                    raise ValueError(f"{step}: unfold a joint before its range is measured.")
                unfolded.add(step.joint)
            elif isinstance(step, Sweep):
                if step.first not in FIRST_ENDS:
                    raise ValueError(f"{step}: first must be one of {FIRST_ENDS}.")
                for joint in step.joints:
                    known(joint, step)
                    if joint in measured:
                        raise ValueError(f"{step}: '{joint}' is measured twice.")
                    if step.first in ("fold", "unfold") and joint not in unfolded:
                        raise ValueError(f"{step}: '{joint}' has no unfold direction yet.")
                measured.update(step.joints)
            elif isinstance(step, Move):
                for joint, target in step.targets.items():
                    known(joint, step)
                    if target == "start":
                        continue
                    if target == "unfolded":
                        if joint not in unfolded:
                            raise ValueError(f"{step}: '{joint}' has no unfold step for target 'unfolded'.")
                        continue
                    if isinstance(target, str) and target not in TARGETS:
                        raise ValueError(f"{step}: unknown target {target!r} for '{joint}'.")
                    if joint not in measured:
                        raise ValueError(f"{step}: '{joint}' needs its range measured for target {target!r}.")
                    if (
                        target in ("fold", "unfold") or not isinstance(target, str)
                    ) and joint not in unfolded:
                        raise ValueError(f"{step}: '{joint}' has no unfold direction for target {target!r}.")
            else:
                raise ValueError(f"Unknown step {step!r}.")
        if missing := motors - measured:
            raise ValueError(f"The plan never measures {sorted(missing)}.")


# ---------------------------------------------------------------------------------------------------------
# Decisions, kept free of bus I/O so they can be tested on their own
# ---------------------------------------------------------------------------------------------------------


def judge_move(
    start: int, goal: int, position: int, load: int, torque_limit: int, cfg: AutoCalibrationConfig
) -> str:
    """Verdict on a position-mode move that has come to rest: "reached", "sagged" or "blocked".

    "sagged" is a joint that stopped short while pushing well below its torque limit: the proportional
    position loop gives way under the arm's weight (on an SO-101 elbow at P=16 by up to 6°), and nothing is
    in the way. A joint stopped short at the torque limit, or before half of the way, is blocked.
    """
    if abs(goal - position) <= deg_to_steps(cfg.move_tolerance_deg):
        return "reached"
    progress = (position - start) / (goal - start) if goal != start else 1.0
    if progress >= cfg.min_progress and abs(load) < cfg.blocked_load * torque_limit:
        return "sagged"
    return "blocked"


class StallDetector:
    """Tells when a joint pushed towards an end stop has come to rest against it.

    The joint follows a goal that runs ahead of it. It rests against a stop when it has lagged its goal by more
    than `lag` steps for `window` seconds and moved less than `progress` steps in that time. The clock starts
    only once the lag is that large: a joint held against a stop by the arm's weight needs its goal several
    degrees ahead before it starts to move away, and a joint that sags under a load moves on.
    """

    def __init__(self, lag: float, progress: float, window: float):
        self.lag, self.progress, self.window = lag, progress, window
        self._since: tuple[float, int] | None = None

    def update(self, t: float, position: int, lag: float) -> bool:
        """`position` in steps travelled (unwrapped), `lag` how far the goal is ahead in the push direction."""
        if lag <= self.lag:
            self._since = None
            return False
        if self._since is None or abs(position - self._since[1]) >= self.progress:
            self._since = (t, position)
            return False
        return t - self._since[0] >= self.window


@dataclass
class JointResult:
    """What the run found out about one joint. Positions are raw encoder steps (Homing_Offset 0)."""

    start: int
    # +1 or -1: the direction of Present_Position in which the joint unfolds from the start pose.
    unfold_sign: int | None = None
    # Where the Unfold step took the joint.
    unfolded: int | None = None
    # The end at the low side of the range and the steps from there up to the other end.
    low: int | None = None
    span: int | None = None
    full_turn: bool = False
    stops: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)

    def ends(self) -> tuple[int, int]:
        """(low, span) of a measured joint with end stops."""
        if self.low is None or self.span is None:
            raise AutoCalibrationError("The joint's range has not been measured.")
        return self.low, self.span

    @property
    def high(self) -> int:
        low, span = self.ends()
        return (low + span) % FULL_TURN

    @property
    def homing_offset(self) -> int:
        """The offset that puts the middle of the range (or, without end stops, the start) at HALF_TURN."""
        if self.full_turn:
            return homing_offset_for(self.start)
        low, span = self.ends()
        return homing_offset_for((low + span // 2) % FULL_TURN)

    def calibration(self, motor_id: int) -> MotorCalibration:
        offset = self.homing_offset
        if self.full_turn:
            return MotorCalibration(motor_id, 0, offset, 0, FULL_TURN - 1)
        low, span = self.ends()
        range_min = (low - offset) % FULL_TURN
        return MotorCalibration(motor_id, 0, offset, range_min, min(range_min + span, FULL_TURN - 1))


# ---------------------------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------------------------


class FeetechAutoCalibrator:
    """Runs an :class:`AutoCalibrationPlan` on a connected Feetech bus and returns the calibration.

    ``run()`` leaves torque off and does not write the calibration to the servos' EEPROM: the robot or
    teleoperator does that and saves the file, as after a manual calibration. `clock` and `sleep` exist for
    tests.
    """

    def __init__(
        self,
        bus: "FeetechMotorsBus",
        plan: AutoCalibrationPlan,
        config: AutoCalibrationConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.bus = bus
        self.plan = plan
        self.cfg = config or AutoCalibrationConfig()
        self._clock = clock
        self._sleep = sleep
        self.names = list(bus.motors)
        self.joints: dict[str, JointResult] = {}
        # Homing_Offset each servo has while the run goes on: first one that puts the start pose at HALF_TURN
        # (moves near the start never cross the encoder's wrap), then, once a joint is measured, the final one.
        self.offsets: dict[str, int] = {}
        self.torque_limits: dict[str, int] = {}

    # --- bus access -----------------------------------------------------------------------------------

    def _read(self, register: str, names: list[str]) -> dict[str, int]:
        return self.bus.sync_read(register, names, normalize=False, num_retry=self.cfg.num_retry)

    def _write(self, register: str, values: dict[str, int]) -> None:
        if values:
            self.bus.sync_write(register, values, normalize=False, num_retry=self.cfg.num_retry)

    def _set(self, name: str, register: str, value: int) -> None:
        self.bus.write(register, name, value, normalize=False, num_retry=self.cfg.num_retry)

    def _raw(self, name: str, present: int) -> int:
        return (present + self.offsets[name]) % FULL_TURN

    # --- run --------------------------------------------------------------------------------------------

    def run(self) -> dict[str, MotorCalibration]:
        self.plan.validate(self.names)
        saved = {
            register: self._read(register, self.names)
            for register in (*RESTORED_REGISTERS, *CALIBRATION_REGISTERS, "Max_Torque_Limit")
        }
        succeeded = False
        try:
            self._prepare(saved)
            for i, step in enumerate(self.plan.steps, 1):
                logger.info(f"Auto-calibration step {i}/{len(self.plan.steps)}: {step}")
                if isinstance(step, Unfold):
                    self._unfold(step)
                elif isinstance(step, Move):
                    self._move_step(step)
                else:
                    self._sweep(step)
            calibration = self._calibration()
            succeeded = True
        finally:
            failures = self._finish(saved, keep_calibration=succeeded)
        if failures:
            # The run's register changes were made with Lock=1, so they live in RAM only.
            raise AutoCalibrationError(
                "The calibration was measured, but restoring the servos failed, so it was not written:\n  "
                + "\n  ".join(failures)
                + "\nSwitch the arm's power off and on to bring back its previous settings."
            )
        return calibration

    def _prepare(self, saved: dict[str, dict[str, int]]) -> None:
        for name in self.names:
            self._set(name, "Torque_Enable", 0)
            # A servo left in velocity mode by earlier software can read Operating_Mode 0 and still ignore
            # Goal_Position; switching to velocity mode and back with torque off ends that.
            self._set(name, "Operating_Mode", 1)
            self._set(name, "Operating_Mode", POSITION_MODE)
            self._set(name, "Lock", 1)
            self._set(name, "Homing_Offset", 0)
            self._set(name, "Min_Position_Limit", 0)
            self._set(name, "Max_Position_Limit", FULL_TURN - 1)
        raw = self._read("Present_Position", self.names)
        torque_limit = self.plan.torque_limit if self.cfg.torque_limit is None else self.cfg.torque_limit
        for name in self.names:
            self.joints[name] = JointResult(start=raw[name])
            self.offsets[name] = homing_offset_for(raw[name])
            self.torque_limits[name] = min(torque_limit, saved["Max_Torque_Limit"][name])
            self._set(name, "Homing_Offset", self.offsets[name])
            self._set(name, "Operating_Mode", POSITION_MODE)
            self._set(name, "Acceleration", self.cfg.acceleration)
            self._set(name, "Torque_Limit", self.torque_limits[name])
            # No speed cap: the setpoint follows the goals the engine steps (see the module docstring).
            self._set(name, "Goal_Velocity", 0)
            if self.cfg.p_coefficient is not None:
                self._set(name, "P_Coefficient", self.cfg.p_coefficient)
        present = self._read("Present_Position", self.names)
        if off := {n: p for n, p in present.items() if abs(p - HALF_TURN) > 2}:
            raise AutoCalibrationError(f"Homing_Offset did not bring the start pose to {HALF_TURN}: {off}")
        # Goal = present before torque on, so nothing moves when torque comes on.
        self._set_goals(present)
        for name in self.names:
            self._set(name, "Torque_Enable", 1)
        self._sleep(self.cfg.start_delay_s)
        now = self._read("Present_Position", self.names)
        tolerance = deg_to_steps(self.cfg.move_tolerance_deg)
        if moved := {n: now[n] - present[n] for n in self.names if abs(now[n] - present[n]) > tolerance}:
            raise AutoCalibrationError(
                f"Joints moved when torque came on, although each goal was its present position: {moved}. "
                "Switch the arm's power off and on again, then retry."
            )
        logger.info(f"Start pose (raw): {raw}")

    def _finish(self, saved: dict[str, dict[str, int]], keep_calibration: bool) -> list[str]:
        """Stop, hold, torque off, and write back what the run changed. Returns what failed.

        Every part runs even if one fails, and a Ctrl-C during the cleanup is raised only once it is done. The
        calibration registers keep the run's result only when the run succeeded and the rest of the cleanup did.
        """
        failures: list[str] = []
        interrupted = False

        def attempt(what: str, action: Callable[[], None]) -> None:
            nonlocal interrupted
            try:
                action()
            except KeyboardInterrupt:
                interrupted = True
                failures.append(f"{what}: interrupted")
                logger.error(
                    f"Auto-calibration cleanup: {what} interrupted; restoring the other registers first"
                )
            except Exception as e:  # noqa: BLE001 - the cleanup has to go on
                failures.append(f"{what}: {e}")
                logger.error(f"Auto-calibration cleanup: {what} failed: {e}")

        attempt("holding", lambda: self._write("Goal_Position", self._read("Present_Position", self.names)))
        for name in self.names:
            attempt(f"torque off on {name}", partial(self._set, name, "Torque_Enable", 0))

        def restore(registers: tuple[str, ...]) -> None:
            for register in registers:
                for name in self.names:
                    attempt(
                        f"restoring {register} on {name}",
                        partial(self._set, name, register, saved[register][name]),
                    )

        restore(RESTORED_REGISTERS)
        if not keep_calibration or failures:
            restore(CALIBRATION_REGISTERS)
        attempt("disabling torque", lambda: self.bus.disable_torque(num_retry=self.cfg.num_retry))
        if interrupted:
            raise KeyboardInterrupt
        return failures

    # --- steps --------------------------------------------------------------------------------------------

    def _unfold(self, step: Unfold) -> None:
        name = step.joint
        origin = self._read("Present_Position", [name])[name]
        if self._move({name: origin + step.sign * deg_to_steps(step.deg)})[name] == "blocked":
            self._move({name: origin})
            raise AutoCalibrationError(
                f"{name} is blocked on its way to unfold. Is the arm in the start pose?\n{self.plan.start_pose}"
            )
        self.joints[name].unfold_sign = step.sign
        self.joints[name].unfolded = self._raw(name, self._read("Present_Position", [name])[name])

    def _move_step(self, step: Move) -> None:
        goals = {name: self._target(name, target) for name, target in step.targets.items()}
        for name, verdict in self._move(goals).items():
            if verdict == "blocked":
                raise AutoCalibrationError(f"{name} was blocked on its way to {step.targets[name]!r}.")

    def _target(self, name: str, target: str | float) -> int:
        """Goal in the servo's present frame for a Move target."""
        joint, offset = self.joints[name], self.offsets[name]
        if target == "start":
            return (joint.start - offset) % FULL_TURN
        if target == "unfolded":
            if joint.unfolded is None:
                raise AutoCalibrationError(f"{name} has not been unfolded.")
            return (joint.unfolded - offset) % FULL_TURN
        if joint.full_turn:
            if target == "mid":
                return (joint.start - offset) % FULL_TURN
            raise AutoCalibrationError(f"{name} has no end stops, so it cannot move to {target!r}.")
        # Once measured, the joint's range sits in the middle of the encoder, so it never crosses the wrap.
        raw_low, span = joint.ends()
        low = (raw_low - offset) % FULL_TURN
        high = low + span
        margin = deg_to_steps(self.cfg.end_margin_deg)
        fold_is_low = joint.unfold_sign is not None and joint.unfold_sign > 0
        if target == "low":
            return low + margin
        if target == "high":
            return high - margin
        if target == "mid":
            return (low + high) // 2
        if target == "fold":
            return low + margin if fold_is_low else high - margin
        if target == "unfold":
            return high - margin if fold_is_low else low + margin
        if isinstance(target, str):
            raise AutoCalibrationError(f"Unknown target {target!r} for {name}.")
        goal = low + deg_to_steps(target) if fold_is_low else high - deg_to_steps(target)
        return max(low + margin, min(high - margin, goal))

    def _move(self, goals: dict[str, int]) -> dict[str, str]:
        """Position-mode move of several joints at once; returns each joint's verdict (see judge_move).

        As in the search for end stops, each joint's goal steps towards its target at the configured velocity,
        never more than three times the stall lag ahead of the joint. A joint that comes to rest against something
        before its goal gets to the target is held there, and is blocked unless it is within the move tolerance
        already; the others are held once they have settled.
        """
        names = list(goals)
        if bad := {n: g for n, g in goals.items() if not 0 <= g < FULL_TURN}:
            raise AutoCalibrationError(f"Goals outside the encoder range: {bad}")
        velocity = self.cfg.velocity
        stall_lag = deg_to_steps(self.cfg.stall_lag_deg)
        progress = deg_to_steps(self.cfg.stall_progress_deg)
        lead = 3 * stall_lag
        start = self._read("Present_Position", names)
        signs = {n: 1 if goals[n] >= start[n] else -1 for n in names}
        goal = {n: float(start[n]) for n in names}
        stalls = {n: StallDetector(stall_lag, progress, self.cfg.stall_window_s) for n in names}
        # Once its goal is at the target, a joint has settled when it moved less than `progress` for a window.
        settles = {n: StallDetector(-1, progress, self.cfg.stall_window_s) for n in names}
        timeout = max(abs(goals[n] - start[n]) for n in names) / velocity * 1.5 + 2.0
        # Where each joint came to rest, its load there (before it was held), and whether it stopped short.
        ends: dict[str, tuple[int, int, bool]] = {}

        t0 = t_last = self._clock()
        running = list(names)
        while running:
            now = self._clock()
            dt, t_last = now - t_last, now
            present = self._read("Present_Position", running)
            timed_out = now - t0 > timeout
            short = {}
            for name in running:
                if goal[name] == goals[name]:
                    if settles[name].update(now, present[name], 0) or timed_out:
                        short[name] = timed_out
                elif (
                    stalls[name].update(now, present[name], (goal[name] - present[name]) * signs[name])
                    or timed_out
                ):
                    short[name] = True
            if short:
                loads = self._read("Present_Load", list(short))
                for name, stopped_short in short.items():
                    ends[name] = (present[name], loads[name], stopped_short)
                self._hold(list(short))
                running = [n for n in running if n not in short]
            for name in running:
                sign = signs[name]
                ahead = min((goal[name] + sign * velocity * dt - present[name]) * sign, lead)
                stepped = present[name] + sign * ahead
                goal[name] = float(goals[name]) if (stepped - goals[name]) * sign >= 0 else stepped
            if running:
                self._write("Goal_Position", {n: round(goal[n]) for n in running})
                self._sleep(self.cfg.poll_interval_s)

        verdicts = {}
        for name in names:
            position, load, stopped_short = ends[name]
            verdict = judge_move(start[name], goals[name], position, load, self.torque_limits[name], self.cfg)
            if stopped_short and verdict != "reached":
                verdict = "blocked"
            verdicts[name] = verdict
            logger.info(
                f"  {name}: goal {goals[name]}, at {position} "
                f"({steps_to_deg(position - goals[name]):+.1f}°), load {load / 10:.0f}%: {verdict}"
            )
        return verdicts

    def _sweep(self, step: Sweep) -> None:
        names = list(step.joints)
        first = {name: self._first_sign(name, step.first) for name in names}
        found = self._drive(names, first)
        # From one end to the other: a joint that stops short of the smallest range the plan allows is blocked. A
        # joint without end stops turns the full turn back, so that it is not left wound up (and its cables with it).
        shortest = {
            n: FULL_TURN if self.joints[n].full_turn else deg_to_steps(self.plan.joints[n].min_deg)
            for n in names
            if self.joints[n].full_turn or self.cfg.check_ranges
        }
        # Travel counts from the first end: a joint can settle back a little from it while others still search
        # (a gripper did by 26 steps).
        origin = {n: found[n][0] for n in names}
        found_back = self._drive(names, {n: -first[n] for n in names}, shortest, origin)
        for name in names:
            self._conclude(name, found[name], found_back[name])

    def _first_sign(self, name: str, first: str) -> int:
        if first in ("positive", "negative"):
            return 1 if first == "positive" else -1
        sign = self.joints[name].unfold_sign
        if sign is None:
            raise AutoCalibrationError(f"{name} has no unfold direction.")
        return sign if first == "unfold" else -sign

    def _drive(
        self,
        names: list[str],
        signs: dict[str, int],
        shortest: dict[str, int] | None = None,
        origin: dict[str, int] | None = None,
    ) -> dict[str, tuple[int, int, str]]:
        """Push the joints towards their end stops until each one rests against one (see StallDetector).

        Position mode: each joint's goal runs ahead of it at the configured velocity, at most three times the stall
        lag ahead, so the push against a stop is what the position loop makes of that lag, and a goal left behind by
        a dead process is a few degrees ahead at most. A joint about to run out of encoder range in its present
        frame gets a new Homing_Offset on the way. Returns {joint: (raw position, steps travelled, how the end was
        found)}. A joint that turns a full turn without stalling is stopped there (and marked full_turn when the
        plan allows it). A joint that stalls before travelling its `shortest` distance stops the whole drive: it
        is blocked, and the others would go on in a pose the plan did not intend. Travel counts from the raw
        positions in `origin`, if given, else from where the joints are.
        """
        velocity = self.cfg.velocity
        held = [n for n in self.names if n not in names]
        check_contact = self.cfg.contact_check != "off" and bool(held)
        stall_lag = deg_to_steps(self.cfg.stall_lag_deg)
        lead = 3 * stall_lag
        # Room left before a goal would pass 0 or 4095; below it the joint is reframed to this far from the edge.
        edge = lead + deg_to_steps(10)
        reframed_distance = HALF_TURN - edge
        present = self._read("Present_Position", names)
        last = {n: self._raw(n, present[n]) for n in names}
        # Steps travelled (unwrapped), and the goal in the same terms.
        travel = {n: wrap(last[n] - origin[n]) if origin else 0 for n in names}
        goal = {n: float(travel[n]) for n in names}
        detectors = {
            n: StallDetector(stall_lag, deg_to_steps(self.cfg.stall_progress_deg), self.cfg.stall_window_s)
            for n in names
        }
        history: deque[tuple[float, dict[str, int]]] = deque(maxlen=200)
        found: dict[str, tuple[int, int, str]] = {}

        timeout = FULL_TURN / velocity * 1.25 + 2.0
        t0 = t_last = self._clock()
        running = list(names)
        while running:
            now = self._clock()
            if now - t0 > timeout:
                self._hold(running)
                progress = ", ".join(f"{n} {steps_to_deg(abs(travel[n])):.0f}°" for n in running)
                raise AutoCalibrationError(f"No end stop within {timeout:.0f} s ({progress}).")
            dt, t_last = now - t_last, now
            present = self._read("Present_Position", running)
            if check_contact:
                history.append((now, self._read("Present_Load", held)))
            stopped = []
            for name in running:
                raw = self._raw(name, present[name])
                travel[name] += wrap(raw - last[name])
                last[name] = raw
                if abs(travel[name]) >= FULL_TURN:
                    if not self.plan.joints[name].full_turn_ok:
                        self._hold(running)
                        raise AutoCalibrationError(f"{name} turned a full turn without meeting an end stop.")
                    self.joints[name].full_turn = True
                    found[name] = (raw, travel[name], "full turn")
                    stopped.append(name)
                elif detectors[name].update(now, travel[name], (goal[name] - travel[name]) * signs[name]):
                    if shortest and abs(travel[name]) < shortest.get(name, 0):
                        self._hold(running)
                        raise AutoCalibrationError(
                            f"{name} stopped after {steps_to_deg(travel[name]):+.1f}°, short of the "
                            f"{steps_to_deg(shortest[name]):.0f}° its range needs: something blocks it."
                        )
                    found[name] = (raw, travel[name], "stall")
                    stopped.append(name)
            if stopped:
                self._hold(stopped)
                for name in stopped:
                    raw, moved, how = found[name]
                    logger.info(f"  {name}: end stop ({how}) at raw {raw} after {steps_to_deg(moved):+.1f}°")
                    if check_contact:
                        self._check_contact(name, history)
                running = [n for n in running if n not in stopped]
            goals = {}
            for name in running:
                sign = signs[name]
                ahead = min((goal[name] + sign * velocity * dt - travel[name]) * sign, lead)
                goal[name] = travel[name] + sign * ahead
                if (sign > 0 and present[name] > FULL_TURN - 1 - edge) or (sign < 0 and present[name] < edge):
                    self._reframe(name, homing_offset_for(last[name], HALF_TURN - sign * reframed_distance))
                    # The joint stopped for it: the goal starts again from where it stands.
                    raw = self._raw(name, self._read("Present_Position", [name])[name])
                    travel[name] += wrap(raw - last[name])
                    last[name] = raw
                    goal[name] = float(travel[name])
                    present[name] = (raw - self.offsets[name]) % FULL_TURN
                goals[name] = max(0, min(FULL_TURN - 1, present[name] + round(goal[name] - travel[name])))
            self._write("Goal_Position", goals)
            if running and logger.isEnabledFor(logging.DEBUG):
                load = self._read("Present_Load", running)
                logger.debug(
                    f"  drive t={now - t0:.2f}s "
                    + ", ".join(
                        f"{n} at {present[n]} goal {goals[n]} lag {(goal[n] - travel[n]) * signs[n]:.0f} "
                        f"load {load[n] / 10:.0f}%"
                        for n in running
                    )
                )
            if running:
                self._sleep(self.cfg.poll_interval_s)
        return found

    def _set_goals(self, goals: dict[str, int]) -> None:
        """Write goals one servo at a time, each write acknowledged, and read them back.

        A sync write gets no reply, so a lost one would leave a joint with a stale goal unnoticed.
        """
        for name, goal in goals.items():
            self._set(name, "Goal_Position", goal)
        if wrong := {n: g for n, g in self._read("Goal_Position", list(goals)).items() if g != goals[n]}:
            raise AutoCalibrationError(f"Goal_Position did not take: {wrong}, written {goals}")

    def _hold(self, names: list[str]) -> None:
        """Make joints hold where they are, and stop pushing at once (with Goal_Velocity 0 the setpoint follows)."""
        self._set_goals(self._read("Present_Position", names))
        self._clear_overload(names)

    def _clear_overload(self, names: list[str]) -> None:
        status = self._read("Status", names)
        for name in names:
            if status[name] & STATUS_OVERLOAD:
                logger.info(f"  {name}: clearing the overload bit (Status 0x{status[name]:02X})")
                self._set(name, "Torque_Enable", 0)
                self._sleep(0.3)
                self._set(name, "Goal_Position", self._read("Present_Position", [name])[name])
                self._set(name, "Torque_Enable", 1)

    def _check_contact(self, name: str, history: deque[tuple[float, dict[str, int]]]) -> None:
        """Compare the held joints' load at the stop with their load 0.4 to 1 s before it."""
        now = history[-1][0]
        before = [loads for t, loads in history if now - 1.0 <= t <= now - 0.4]
        if len(before) < 3:
            logger.info(f"  {name}: stopped too soon after starting for a contact check")
            return
        at_stop = [loads for _, loads in list(history)[-2:]]
        for other in at_stop[-1]:
            load_before = statistics.median(loads[other] for loads in before)
            load_at_stop = statistics.median(loads[other] for loads in at_stop)
            if abs(load_at_stop - load_before) >= self.cfg.contact_load_delta:
                change = f"{other} load {load_before / 10:.0f}% -> {load_at_stop / 10:.0f}%"
                message = (
                    f"{name} stopped while {change}: the arm may be pushing on something (table, clamp) "
                    f"instead of resting against {name}'s own end stop."
                )
                self.joints[name].contacts.append(f"contact? {change}")
                if self.cfg.contact_check == "abort":
                    raise AutoCalibrationError(message)
                logger.warning(message)

    def _conclude(self, name: str, first: tuple[int, int, str], second: tuple[int, int, str]) -> None:
        joint = self.joints[name]
        joint.stops = [first[2], second[2]]
        if joint.full_turn:
            logger.info(f"  {name}: no end stops, full encoder range")
            return
        (raw_first, _, _), (raw_second, travelled, _) = first, second
        span = joint.span = abs(travelled)
        low = joint.low = raw_second if travelled < 0 else raw_first
        other_end = raw_first if travelled < 0 else raw_second
        if abs(wrap(low + span - other_end)) > 10:
            raise AutoCalibrationError(f"{name}: tracked travel does not match the end positions.")
        from_low = (joint.start - low) % FULL_TURN
        if from_low > span + deg_to_steps(2):
            raise AutoCalibrationError(f"{name}: the start position lies outside the measured range.")
        if joint.unfold_sign is not None:
            to_fold_end = from_low if joint.unfold_sign > 0 else span - from_low
            if to_fold_end > span / 2:
                raise AutoCalibrationError(
                    f"{name} started nearer its unfold end than its fold end: the start pose was not folded, "
                    "or the unfold went the wrong way."
                )
        logger.info(
            f"  {name}: {steps_to_deg(span):.1f}° between raw {low} and {joint.high}, "
            f"start {steps_to_deg(from_low):.1f}° from the low end"
        )
        self._reframe(name, joint.homing_offset)

    def _reframe(self, name: str, offset: int) -> None:
        """Give a joint a new Homing_Offset without moving it.

        The joint is stopped first. Its goal has to move with its frame; it is written right after the offset, a
        few milliseconds in which the servo's acceleration limit moves the joint by well under a step.
        """
        if offset == self.offsets[name]:
            return
        self._hold([name])
        t0 = self._clock()
        while self._read("Moving", [name])[name] and self._clock() - t0 < 1.0:
            self._sleep(self.cfg.poll_interval_s)
        raw = self._raw(name, self._read("Present_Position", [name])[name])
        self._set(name, "Homing_Offset", offset)
        self.offsets[name] = offset
        goal = (raw - offset) % FULL_TURN
        self._set_goals({name: goal})
        present = self._read("Present_Position", [name])[name]
        if abs(wrap(present - goal)) > deg_to_steps(2):
            raise AutoCalibrationError(f"{name}: at {present} after Homing_Offset {offset}, expected {goal}.")

    def _calibration(self) -> dict[str, MotorCalibration]:
        problems = []
        for name, joint in self.joints.items():
            expected = self.plan.joints[name]
            if joint.full_turn or not self.cfg.check_ranges:
                continue
            deg = steps_to_deg(joint.ends()[1])
            if not expected.min_deg <= deg <= expected.max_deg:
                problems.append(f"{name}: {deg:.1f}°, expected {expected.min_deg:g} to {expected.max_deg:g}°")
        if problems:
            raise AutoCalibrationError(
                "Implausible ranges, nothing was written:\n  "
                + "\n  ".join(problems)
                + "\nSomething may have stopped these joints before their end stops. If nothing did, and the arm's "
                "ranges really are these, run again with --auto_calibration.check_ranges=false."
            )
        return {name: joint.calibration(self.bus.motors[name].id) for name, joint in self.joints.items()}

    def summary(self) -> str:
        """One line per joint: range, ends and how they were found, contacts."""
        lines = [f"{'joint':<14} {'range':>8} {'low':>5} {'high':>5}  ends found by"]
        for name, joint in self.joints.items():
            if joint.full_turn:
                lines.append(f"{name:<14} {'360.0°':>8} {'-':>5} {'-':>5}  no end stops")
            elif joint.span is not None:
                low, span = joint.ends()
                ends = ", ".join(joint.stops + joint.contacts)
                lines.append(f"{name:<14} {steps_to_deg(span):>7.1f}° {low:>5} {joint.high:>5}  {ends}")
        return "\n".join(lines)


def auto_calibrate(
    bus: "FeetechMotorsBus",
    plan: AutoCalibrationPlan,
    config: AutoCalibrationConfig | None = None,
) -> dict[str, MotorCalibration]:
    """Run `plan` on `bus` (see FeetechAutoCalibrator) and log a summary of what was measured."""
    calibrator = FeetechAutoCalibrator(bus, plan, config)
    try:
        return calibrator.run()
    finally:
        logger.info("Auto-calibration results:\n" + calibrator.summary())
