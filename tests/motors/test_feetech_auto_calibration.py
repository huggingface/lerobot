#!/usr/bin/env python

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

"""Automatic calibration, run against a simulated SO-101 arm.

`SimArm` stands in for a FeetechMotorsBus: six servos with end stops, driven in position mode (velocity mode too,
which the calibration must never switch on with torque on), reporting position, velocity, Moving, Status and load.
Time only advances when the calibrator sleeps.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.auto_calibration import (
    FULL_TURN,
    HALF_TURN,
    AutoCalibrationConfig,
    AutoCalibrationError,
    AutoCalibrationPlan,
    FeetechAutoCalibrator,
    JointRange,
    JointResult,
    Move,
    StallDetector,
    Sweep,
    Unfold,
    deg_to_steps,
    homing_offset_for,
    judge_move,
)
from lerobot.robots.so_follower.auto_calibration_plan import SO_ARM_PLAN, SO_LEADER_PLAN

RESTORED = ("Operating_Mode", "Torque_Limit", "Acceleration", "Goal_Velocity", "P_Coefficient")
CALIBRATION = ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")


class SimJoint:
    """One servo on a joint with end stops at raw positions `low` < `high` (unwrapped, may pass 4095).

    In position mode the servo's setpoint travels towards Goal_Position at Goal_Velocity (0: at once), in raw
    encoder terms, and the joint follows it at up to `max_speed`. As on firmware 3.10, the speed is the one
    Goal_Velocity had when the setpoint started to move, until it arrives: a speed written on the way is ignored.
    Against a stop the setpoint runs on and the joint pushes with a load that grows with the gap, 0.4 % per step up
    to Torque_Limit, as an STS3215 at P=16 does.
    """

    def __init__(
        self,
        low,
        high,
        start,
        *,
        free=False,
        sag=0,
        max_speed=3000,
        contact_at_high=None,
        runaway=0,
        breakaway=0,
    ):
        self.low, self.high, self.pos = float(low), float(high), float(start)
        self.setpoint = self.pos
        self.free = free  # no end stops at all
        self.sag = sag  # steps a downward position move stops short, as a loaded position loop does
        self.max_speed = max_speed
        self.contact_at_high = contact_at_high or {}  # load on other joints while pushing at `high`
        # Speed at which the servo runs whenever its torque is on, whatever its goal: a servo left in velocity mode
        # although Operating_Mode reads 0, as seen on firmware 3.10.
        self.runaway = runaway
        # Gap between setpoint and joint a joint that has rested for a while needs before it starts to move, as
        # when the arm's weight holds it. A moving joint follows its setpoint.
        self.breakaway = breakaway
        self.resting = 0.0
        # Speed the setpoint travels at, taken from Goal_Velocity when it starts to move; None while it rests.
        self.speed = None
        self.velocity = 0.0
        self.pushing = 0
        self.load = 0
        self.regs = {
            "Torque_Enable": 1,
            "Lock": 1,
            "Homing_Offset": 25,
            "Min_Position_Limit": 700,
            "Max_Position_Limit": 3400,
            "Operating_Mode": 0,
            "Torque_Limit": 1000,
            "Max_Torque_Limit": 1000,
            "Acceleration": 254,
            "Goal_Velocity": 0,
            "Goal_Position": 0,
            "P_Coefficient": 16,
            "Status": 0,
        }

    def present(self) -> int:
        return (round(self.pos) - self.regs["Homing_Offset"]) % FULL_TURN

    def step(self, dt: float) -> None:
        r = self.regs
        if not r["Torque_Enable"]:
            self.velocity, self.pushing, self.load = 0.0, 0, 0
            self.setpoint, self.speed = self.pos, None
            return
        gap = None
        if self.runaway:
            v = self.runaway
        elif r["Operating_Mode"] == 1:
            v = r["Goal_Velocity"]
        else:
            goal = min(max(r["Goal_Position"], r["Min_Position_Limit"]), r["Max_Position_Limit"])
            present = self.present()
            if goal < present - self.sag:
                goal += self.sag
            target = self.pos + goal - present  # the servo moves within its frame, never across 0/4095
            if self.speed is None:
                self.speed = r["Goal_Velocity"]
            limit = self.speed * dt if self.speed else float("inf")
            self.setpoint += max(-limit, min(limit, target - self.setpoint))
            if self.setpoint == target:
                self.speed = None
            gap = self.setpoint - self.pos
            at_rest = self.resting >= 0.1 and abs(gap) < self.breakaway
            v = 0.0 if at_rest else max(-self.max_speed, min(self.max_speed, gap / dt))
        new = self.pos + v * dt
        self.pushing = 0
        if not self.free and new >= self.high and v > 0:
            new, self.pushing = self.high, 1
        elif not self.free and new <= self.low and v < 0:
            new, self.pushing = self.low, -1
        self.velocity = 0.0 if self.pushing else (new - self.pos) / dt
        self.resting = self.resting + dt if self.velocity == 0 else 0.0
        self.pos = new
        if self.pushing:
            push = r["Torque_Limit"] if gap is None else min(r["Torque_Limit"], 4 * abs(gap))
            self.load = self.pushing * push
        else:
            self.load = 40


class SimArm:
    """The part of FeetechMotorsBus the calibrator uses, over simulated joints."""

    def __init__(self, joints: dict[str, SimJoint], interrupt_at: float | None = None):
        self.joints = joints
        # (joint, register) writes to drop silently, e.g. a lost packet
        self.dropped: list[tuple[str, str]] = []
        # (joint, register, value, exception) writes that raise once, e.g. a dead bus or a Ctrl-C
        self.failing: list[tuple[str, str, int, type[BaseException]]] = []
        self.motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(joints, 1)}
        self.t = 0.0
        self.interrupt_at = interrupt_at
        self.writes: list[tuple[float, str, str, int]] = []
        self.initial = {n: dict(j.regs) for n, j in joints.items()}

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        end = self.t + dt
        while self.t < end - 1e-9:
            step = min(0.005, end - self.t)
            for joint in self.joints.values():
                joint.step(step)
            self.t += step
        if self.interrupt_at is not None and self.t >= self.interrupt_at:
            raise KeyboardInterrupt

    def _get(self, name: str, register: str) -> int:
        joint = self.joints[name]
        if register == "Present_Position":
            return joint.present()
        if register == "Present_Velocity":
            return round(joint.velocity)
        if register == "Moving":
            # Set while the setpoint travels, also for a joint that stands against a stop (firmware 3.10).
            return int(joint.speed is not None or abs(joint.velocity) > 0.5)
        if register == "Present_Load":
            load = joint.load
            for other in self.joints.values():
                if other.pushing > 0 and name in other.contact_at_high:
                    load += other.contact_at_high[name]
            return load
        return joint.regs[register]

    def _put(self, name: str, register: str, value: int) -> None:
        for failure in self.failing:
            if failure[:3] == (name, register, value):
                self.failing.remove(failure)
                raise failure[3](f"writing {register}={value} to {name} failed")
        if (name, register) in self.dropped:
            self.dropped.remove((name, register))
            return
        self.writes.append((self.t, name, register, value))
        self.joints[name].regs[register] = value
        if register == "Goal_Position":  # firmware 3.10: writing a goal switches torque on
            self.joints[name].regs["Torque_Enable"] = 1

    def read(self, register, motor, *, normalize=True, num_retry=0):
        return self._get(motor, register)

    def sync_read(self, register, motors=None, *, normalize=True, num_retry=0):
        return {n: self._get(n, register) for n in (motors or self.joints)}

    def write(self, register, motor, value, *, normalize=True, num_retry=0):
        self._put(motor, register, value)

    def sync_write(self, register, values, *, normalize=True, num_retry=0):
        for name, value in values.items():
            self._put(name, register, value)

    def disable_torque(self, motors=None, num_retry=0):
        for name in motors or self.joints:
            self._put(name, "Torque_Enable", 0)
            self._put(name, "Lock", 0)


# End stops and start pose (raw, Homing_Offset 0) as measured on an SO-101 follower. The arm starts folded:
# shoulder_lift 24° from its fold end, elbow_flex and wrist_flex just short of theirs, gripper closed.
SO101 = {
    "shoulder_pan": (963, 3181, 2072),
    "shoulder_lift": (3000, 5424, 3279),  # the range crosses the encoder's 4095 -> 0 wrap
    "elbow_flex": (1571, 3815, 3787),
    "wrist_flex": (1478, 3830, 3720),
    "wrist_roll": (1749, 5657, 3700),
    "gripper": (1652, 3261, 1660),
}


# Shoulder and elbow start from stops that the arm's weight presses them into (steps of goal lead to break away).
BREAKAWAY = {"shoulder_lift": 60, "elbow_flex": 60}


def make_arm(interrupt_at=None, **overrides) -> SimArm:
    joints = {}
    for name, (low, high, start) in SO101.items():
        kwargs = {"breakaway": BREAKAWAY.get(name, 0), **overrides.get(name, {})}
        low, high, start = kwargs.pop("low", low), kwargs.pop("high", high), kwargs.pop("start", start)
        joints[name] = SimJoint(low, high, start, **kwargs)
    return SimArm(joints, interrupt_at)


def run(arm: SimArm, config: AutoCalibrationConfig | None = None, plan=SO_ARM_PLAN) -> FeetechAutoCalibrator:
    calibrator = FeetechAutoCalibrator(arm, plan, config, clock=arm.clock, sleep=arm.sleep)
    calibrator.result = calibrator.run()
    return calibrator


def assert_restored(arm: SimArm, registers) -> None:
    for name, joint in arm.joints.items():
        for register in registers:
            assert joint.regs[register] == arm.initial[name][register], (name, register)
        assert joint.regs["Torque_Enable"] == 0, name
        assert joint.regs["Lock"] == 0, name


def test_so101_full_run():
    arm = make_arm()
    calibrator = run(arm)

    for name, (low, high, start) in SO101.items():
        joint = calibrator.joints[name]
        assert abs(joint.span - (high - low)) <= 3, name
        assert abs(joint.low - low % FULL_TURN) <= 3, name
        cal = calibrator.result[name]
        assert cal.id == arm.motors[name].id
        assert cal.range_min == (joint.low - cal.homing_offset) % FULL_TURN
        assert cal.range_max == cal.range_min + joint.span
        assert abs((cal.range_min + cal.range_max) / 2 - HALF_TURN) <= 1, name
        # The plan brings the arm back to its start pose.
        assert abs(arm.joints[name].pos % FULL_TURN - start) <= 3, name
        assert not joint.contacts

    assert calibrator.joints["shoulder_lift"].unfold_sign == 1
    assert calibrator.joints["elbow_flex"].unfold_sign == -1
    assert calibrator.joints["wrist_flex"].unfold_sign == -1
    # Registers are back; the calibration registers hold the result's offsets, for the caller to persist.
    assert_restored(arm, RESTORED)
    for name, cal in calibrator.result.items():
        assert arm.joints[name].regs["Homing_Offset"] == cal.homing_offset


def test_wrist_never_turns_up_in_the_folded_arm():
    # Turned up in the folded arm, the gripper (or a camera on the wrist) meets the forearm: a follower's wrist
    # stopped 55 degrees short there. With the elbow folded, the wrist stays between hanging down and forwards.
    arm = make_arm()
    elbow, wrist = arm.joints["elbow_flex"], arm.joints["wrist_flex"]
    folded_wrist = []
    sleep = arm.sleep

    def traced_sleep(dt):
        sleep(dt)
        if elbow.pos > elbow.high - deg_to_steps(20):
            folded_wrist.append(wrist.pos)

    arm.sleep = traced_sleep
    calibrator = run(arm)
    unfolded = calibrator.joints["wrist_flex"].unfolded
    assert abs(unfolded - (SO101["wrist_flex"][2] - deg_to_steps(80))) <= 3
    assert min(folded_wrist) >= unfolded - deg_to_steps(5)


def test_joint_settling_back_between_its_two_ends():
    # A follower's gripper settled 26 steps back from its open stop while the wrist roll still searched: its range
    # is still the distance between its two ends.
    arm = make_arm()
    gripper = arm.joints["gripper"]
    drive = FeetechAutoCalibrator._drive

    def settle_then_drive(self, names, signs, *args):
        if "gripper" in names and args:  # the second drive of the wrist group
            gripper.pos -= 26
            gripper.setpoint = gripper.pos
            self._set_goals({"gripper": gripper.present()})
        return drive(self, names, signs, *args)

    with patch.object(FeetechAutoCalibrator, "_drive", settle_then_drive):
        calibrator = run(arm)
    low, high, _ = SO101["gripper"]
    assert abs(calibrator.joints["gripper"].span - (high - low)) <= 3


def test_goal_is_present_position_before_torque_on():
    arm = make_arm()
    run(arm)
    for name in arm.joints:
        writes = [(reg, value) for _, n, reg, value in arm.writes if n == name]
        first_goal = next(value for reg, value in writes if reg == "Goal_Position")
        torque_on = writes.index(("Torque_Enable", 1))
        assert first_goal == HALF_TURN  # the start pose, in the start-centred frame
        assert writes.index(("Goal_Position", first_goal)) < torque_on


def test_no_joint_pushes_at_its_torque_limit_for_long():
    # A setpoint sent far ahead keeps a blocked joint pushing until it arrives, and a speed written meanwhile does
    # not stop it: on hardware, a wrist sent 80 degrees into its fold stop pushed at the torque limit for about 8 s.
    arm = make_arm()
    pushing = dict.fromkeys(arm.joints, 0.0)
    longest = dict.fromkeys(arm.joints, 0.0)
    for name, joint in arm.joints.items():

        def traced_step(dt, name=name, joint=joint, step=joint.step):
            step(dt)
            if joint.pushing and abs(joint.load) >= 0.9 * joint.regs["Torque_Limit"]:
                pushing[name] += dt
                longest[name] = max(longest[name], pushing[name])
            else:
                pushing[name] = 0.0

        joint.step = traced_step
    run(arm)
    assert max(longest.values()) < 1.0, longest


def test_torque_limit_never_above_max_torque_limit():
    arm = make_arm()
    arm.joints["gripper"].regs["Max_Torque_Limit"] = 500
    arm.initial["gripper"]["Max_Torque_Limit"] = 500
    run(arm)
    limits = [value for _, n, reg, value in arm.writes if n == "gripper" and reg == "Torque_Limit"]
    assert limits[0] == 500


@pytest.mark.parametrize(
    "plan, torque_limit, expected",
    [(SO_ARM_PLAN, None, 500), (SO_LEADER_PLAN, None, 800), (SO_LEADER_PLAN, 600, 600)],
)
def test_torque_limit_from_the_plan_unless_configured(plan, torque_limit, expected):
    arm = make_arm()
    # The simulated arm is a follower, whose gripper opens wider than a leader's trigger.
    run(arm, AutoCalibrationConfig(torque_limit=torque_limit, check_ranges=False), plan)
    for name in arm.joints:
        limits = [value for _, n, reg, value in arm.writes if n == name and reg == "Torque_Limit"]
        assert limits[0] == expected, name


def test_sagging_elbow_still_unfolds():
    # 7° of sag fails a fixed 5° tolerance, but the elbow went most of the way at a low load.
    arm = make_arm(elbow_flex={"sag": deg_to_steps(7)})
    calibrator = run(arm)
    assert calibrator.joints["elbow_flex"].unfold_sign == -1


def test_boxed_in_wrist_aborts_before_any_sweep():
    # Gripper pointing up: the folded arm leaves the wrist 20° of travel (a follower's stopped after 55°).
    arm = make_arm(wrist_flex={"low": 3600, "high": 3830})
    with pytest.raises(AutoCalibrationError, match="wrist_flex is blocked on its way to unfold"):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_joint_stuck_at_its_first_end_stops_the_whole_drive():
    # The shoulder cannot leave its fold stop: the elbow must not go on stretching the arm (on a follower, it hung
    # the straight arm behind the table edge).
    # From the sweep on it needs more breakaway than a goal ever leads by (3 x 4 degrees).
    arm = make_arm()
    sweep = FeetechAutoCalibrator._sweep

    def stick_then_sweep(self, step):
        if "shoulder_lift" in step.joints:
            arm.joints["shoulder_lift"].breakaway = 150
        return sweep(self, step)

    with (
        patch.object(FeetechAutoCalibrator, "_sweep", stick_then_sweep),
        pytest.raises(AutoCalibrationError, match="shoulder_lift stopped after .* something blocks it"),
    ):
        run(arm)
    elbow = arm.joints["elbow_flex"]
    assert elbow.pos > elbow.high - deg_to_steps(30)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_timeout_aborts_and_restores():
    arm = make_arm(shoulder_pan={"max_speed": 30})
    with pytest.raises(AutoCalibrationError, match="No end stop"):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_full_turn_only_where_the_plan_allows_it():
    arm = make_arm(shoulder_pan={"free": True})
    with pytest.raises(AutoCalibrationError, match="full turn"):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_wrist_roll_without_stops_gets_the_full_range():
    arm = make_arm(wrist_roll={"free": True})
    calibrator = run(arm)
    cal = calibrator.result["wrist_roll"]
    assert calibrator.joints["wrist_roll"].full_turn
    assert (cal.range_min, cal.range_max) == (0, FULL_TURN - 1)
    assert cal.homing_offset == homing_offset_for(SO101["wrist_roll"][2])
    # It turned the full turn back, so it is not left wound up (with its cables).
    assert calibrator.joints["wrist_roll"].stops == ["full turn", "full turn"]
    assert abs(arm.joints["wrist_roll"].pos - SO101["wrist_roll"][2]) <= 5


def test_blocked_joint_aborts():
    # The base stopped by a clamp after 43°, as in a run with the arm stretched out behind the base.
    arm = make_arm(shoulder_pan={"low": 1900, "high": 2388})
    with pytest.raises(AutoCalibrationError, match="shoulder_pan stopped after -42.9°, short of the 185°"):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_implausible_range_aborts():
    # A gripper that opens 210°: more than the plan allows.
    arm = make_arm(gripper={"high": 1652 + deg_to_steps(210)})
    with pytest.raises(AutoCalibrationError, match="gripper: 210.0°"):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_joint_stopped_well_before_its_end_stop_is_refused():
    # Something stops the base 30 degrees before its end stop: its range is still more than half a turn.
    low, high, _ = SO101["shoulder_pan"]
    arm = make_arm(shoulder_pan={"high": high - deg_to_steps(30)})
    with pytest.raises(
        AutoCalibrationError, match=r"shoulder_pan stopped after -16\d\.\d°, short of the 185°"
    ):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_implausible_range_accepted_when_the_check_is_off():
    arm = make_arm(shoulder_pan={"low": 1900, "high": 2388})
    calibrator = run(arm, AutoCalibrationConfig(check_ranges=False))
    assert abs(calibrator.joints["shoulder_pan"].span - 488) <= 3


@pytest.mark.parametrize("mode", ["warn", "abort"])
def test_contact_with_the_table_is_noticed(mode):
    # Pressing the stretched arm onto the table loads the held wrist.
    arm = make_arm(shoulder_lift={"contact_at_high": {"wrist_flex": 300}})
    config = AutoCalibrationConfig(contact_check=mode)
    if mode == "abort":
        with pytest.raises(
            AutoCalibrationError, match="shoulder_lift stopped while wrist_flex load 4% -> 34%"
        ):
            run(arm, config)
        assert_restored(arm, RESTORED + CALIBRATION)
    else:
        calibrator = run(arm, config)
        assert len(calibrator.joints["shoulder_lift"].contacts) == 1
        assert not calibrator.joints["elbow_flex"].contacts


def test_ctrl_c_restores_the_registers():
    arm = make_arm(interrupt_at=20.0)
    with pytest.raises(KeyboardInterrupt):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_unfold_goes_only_the_plans_way():
    # An elbow that starts 26 degrees open could also move freely towards its fold stop, and end within the move
    # tolerance of a 30-degree goal there: it unfolds the way the plan says, and the run measures it right.
    low, high, _ = SO101["elbow_flex"]
    arm = make_arm(elbow_flex={"start": high - deg_to_steps(26)})
    calibrator = run(arm)
    elbow = calibrator.joints["elbow_flex"]
    assert elbow.unfold_sign == -1
    assert abs(elbow.span - (high - low)) <= 3
    assert abs(elbow.low - low) <= 3


def test_unfold_blocked_in_the_plans_direction_aborts():
    # A plan with the elbow's unfold direction the wrong way round: the folded elbow meets its fold stop at once.
    steps = tuple(
        replace(step, sign=1) if isinstance(step, Unfold) and step.joint == "elbow_flex" else step
        for step in SO_ARM_PLAN.steps
    )
    arm = make_arm()
    with pytest.raises(AutoCalibrationError, match="elbow_flex is blocked on its way to unfold"):
        run(arm, plan=replace(SO_ARM_PLAN, steps=steps))
    assert_restored(arm, RESTORED + CALIBRATION)


def test_failed_restore_writes_no_calibration():
    # Everything was measured, but one register could not be written back: no result, and the previous
    # calibration is back in the servos.
    arm = make_arm()
    arm.failing.append(("gripper", "Torque_Limit", 1000, ConnectionError))
    with pytest.raises(AutoCalibrationError, match="restoring Torque_Limit on gripper"):
        run(arm)
    gripper = arm.joints["gripper"]
    assert gripper.regs["Torque_Limit"] == 500
    gripper.regs["Torque_Limit"] = 1000  # the failed write aside, everything is restored
    assert_restored(arm, RESTORED + CALIBRATION)


def test_ctrl_c_during_the_cleanup_restores_the_rest_first():
    arm = make_arm()
    arm.failing.append(("shoulder_pan", "Torque_Limit", 1000, KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        run(arm)
    pan = arm.joints["shoulder_pan"]
    assert pan.regs["Torque_Limit"] == 500
    pan.regs["Torque_Limit"] = 1000  # the interrupted write aside, everything is restored
    assert_restored(arm, RESTORED + CALIBRATION)


def test_start_pose_nearer_the_unfold_end_aborts():
    # The shoulder starts stretched forwards: the unfold still moves, but the pose was not folded.
    arm = make_arm(shoulder_lift={"start": 5200})
    with pytest.raises(AutoCalibrationError, match="nearer its unfold end"):
        run(arm)


# --- decisions -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position, load, verdict",
    [
        (2388, 50, "reached"),
        (2340, 50, "reached"),  # 4.2° short
        (2300, 300, "sagged"),  # 7.7° short at 30 % load
        (2300, 790, "blocked"),  # 7.7° short at the torque limit
        (2076, 100, "blocked"),  # 8 % of the way
    ],
)
def test_judge_move(position, load, verdict):
    assert judge_move(2048, 2388, position, load, 800, AutoCalibrationConfig()) == verdict


def test_stall_detector():
    # Far behind its goal but moving on, as a joint sagging under a load: no stall.
    sagging = StallDetector(lag=45, progress=6, window=0.3)
    assert not any(sagging.update(i * 0.05, i * 10, 80) for i in range(20))
    # At rest close to its goal: no stall.
    resting = StallDetector(lag=45, progress=6, window=0.3)
    assert not any(resting.update(i * 0.05, 0, 30) for i in range(20))
    # At rest and far behind: a stall once a whole window has passed.
    stalled = StallDetector(lag=45, progress=6, window=0.3)
    results = [stalled.update(i * 0.05, 100 + i % 2, 60) for i in range(10)]
    assert not any(results[:6]) and all(results[6:])
    # Breaking away from a stop: still while the lag builds up to 60 steps, then moving. The window starts only
    # when the lag passes 45.
    breaking = StallDetector(lag=45, progress=6, window=0.3)
    lags = [i * 10 for i in range(7)] + [60] * 10
    positions = [0] * 7 + [(i + 1) * 8 for i in range(10)]
    assert not any(
        breaking.update(i * 0.05, p, lag) for i, (p, lag) in enumerate(zip(positions, lags, strict=True))
    )


def test_lost_goal_write_aborts_before_torque_on():
    # Goals are written one servo at a time and read back: a lost write stops the run before torque comes on.
    arm = make_arm()
    arm.dropped.append(("elbow_flex", "Goal_Position"))
    with pytest.raises(AutoCalibrationError, match="did not take"):
        run(arm)
    assert not any(reg == "Torque_Enable" and value == 1 for _, _, reg, value in arm.writes)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_joint_moving_at_torque_on_aborts():
    arm = make_arm(elbow_flex={"runaway": -300})
    with pytest.raises(AutoCalibrationError, match="moved when torque came on"):
        run(arm)
    assert_restored(arm, RESTORED + CALIBRATION)


def test_no_velocity_mode():
    arm = make_arm()
    run(arm)
    # Operating_Mode 1 is written only with torque off, to clear a servo stuck in velocity mode.
    torque = dict.fromkeys(arm.joints, 1)
    for _, name, reg, value in arm.writes:
        if reg == "Torque_Enable":
            torque[name] = value
        elif reg == "Goal_Position":
            torque[name] = 1
        elif reg == "Operating_Mode" and value != 0:
            assert torque[name] == 0, name
    # The shoulder's range is wider than the room its start frame leaves: it was reframed on the way.
    offsets = [v for _, n, reg, v in arm.writes if n == "shoulder_lift" and reg == "Homing_Offset"]
    assert len(set(offsets)) >= 3


def test_range_across_the_encoder_wrap():
    joint = JointResult(start=3279, low=3000, span=2424)
    cal = joint.calibration(2)
    assert cal == MotorCalibration(2, 0, homing_offset_for((3000 + 1212) % FULL_TURN), 835, 3259)
    assert joint.high == (3000 + 2424) % FULL_TURN


def test_homing_offset_stays_in_its_register_range():
    assert homing_offset_for(HALF_TURN) == 0
    assert homing_offset_for(HALF_TURN - 1) == -1
    for raw in range(FULL_TURN):
        offset = homing_offset_for(raw)
        assert -2047 <= offset <= 2047
        # Present_Position = Actual - Homing_Offset: within a step of the middle even at +/-2048.
        assert abs((raw - offset) % FULL_TURN - HALF_TURN) <= 1


@pytest.mark.parametrize("plan", [SO_ARM_PLAN, SO_LEADER_PLAN])
def test_so_arm_plans_are_valid(plan):
    plan.validate(SO101)


@pytest.mark.parametrize(
    "steps, message",
    [
        ((Sweep(("a",), first="fold"),), "no unfold direction"),
        ((Move({"a": "low"}), Sweep(("a",))), "needs its range measured"),
        ((Sweep(("a",)), Sweep(("a",))), "measured twice"),
        ((Unfold("a", 10, sign=1),), "never measures"),
        ((Unfold("a", 10, sign=0), Sweep(("a",))), "sign must be"),
        ((Sweep(("a",)), Move({"a": "sideways"})), "unknown target"),
        ((Sweep(("a",)), Move({"a": "unfolded"})), "no unfold step"),
        ((Sweep(("b",)),), "unknown joint"),
    ],
)
def test_plan_validation(steps, message):
    plan = AutoCalibrationPlan(start_pose="", joints={"a": JointRange(0, 360)}, steps=steps)
    with pytest.raises(ValueError, match=message):
        plan.validate(["a"])


# --- SO arms and lerobot-calibrate --------------------------------------------------------------------------


def _fake_result():
    return {name: MotorCalibration(i, 0, 10 * i, 900, 3100) for i, name in enumerate(SO101, 1)}


@pytest.mark.parametrize("kind", ["follower", "leader"])
def test_so_arm_saves_through_its_own_calibration_file(tmp_path, kind):
    bus = MagicMock(name="FeetechBusMock")
    if kind == "follower":
        from lerobot.robots.so_follower import SO101Follower as Device, SO101FollowerConfig as Config

        module = "lerobot.robots.so_follower.so_follower"
    else:
        from lerobot.teleoperators.so_leader import SO101Leader as Device, SO101LeaderConfig as Config

        module = "lerobot.teleoperators.so_leader.so_leader"
    # Whether torque was off when the user was asked to put the arm in the start pose.
    torque_off_at_prompt = []
    with (
        patch(f"{module}.FeetechMotorsBus", return_value=bus),
        patch(f"{module}.auto_calibrate", return_value=_fake_result()) as auto,
        patch(
            "builtins.input", side_effect=lambda *_: torque_off_at_prompt.append(bus.disable_torque.called)
        ),
    ):
        device = Device(Config(port="/dev/null", id="arm", calibration_dir=tmp_path))
        device.auto_calibrate(AutoCalibrationConfig(velocity=200))

    assert torque_off_at_prompt == [True]
    assert auto.call_args.args[0] is bus and auto.call_args.args[2].velocity == 200
    assert auto.call_args.args[1] is (SO_ARM_PLAN if kind == "follower" else SO_LEADER_PLAN)
    bus.write_calibration.assert_called_once_with(_fake_result())
    assert device.calibration_fpath == tmp_path / "arm.json"
    assert device.calibration_fpath.is_file()


def test_lerobot_calibrate_auto():
    from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="no automatic calibration"):
        calibrate(CalibrateConfig(robot=MockRobotConfig(), auto=True))

    device = MagicMock(name="SOFollowerMock")
    cfg = CalibrateConfig(
        robot=MockRobotConfig(), auto=True, auto_calibration=AutoCalibrationConfig(velocity=150)
    )
    with patch("lerobot.scripts.lerobot_calibrate.make_robot_from_config", return_value=device):
        calibrate(cfg)
    device.auto_calibrate.assert_called_once_with(cfg.auto_calibration)
    device.calibrate.assert_not_called()
    device.disconnect.assert_called_once()


def test_lerobot_calibrate_help(capsys):
    """`--help` lists the options: argparse %-formats help text, and a "%" in a field comment broke it."""
    from lerobot.scripts.lerobot_calibrate import calibrate

    with patch("sys.argv", ["lerobot-calibrate", "--help"]), pytest.raises(SystemExit) as exit_info:
        calibrate()
    assert exit_info.value.code == 0
    assert "--auto_calibration.torque_limit" in capsys.readouterr().out
