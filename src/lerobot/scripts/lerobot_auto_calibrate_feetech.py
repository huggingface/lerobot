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

"""
Feetech STS servo auto-calibration (with unfolding).

Full workflow (single command):
  Stage 0  Initialize: stop all servos, Lock=1, configure PID/acceleration, enable torque
  Stage 2  Unfold joints 2-4 (can be skipped with --unfold-angle 0)
  Stage 3  Calibrate servos 2-6 (5->6->4->3->2)
  Stage 4  Finally calibrate servo 1 shoulder_pan and return to center
  Stage 5  Wait for user confirmation then disable torque

Usage examples:

  lerobot-auto-calibrate-feetech --port COM3
  lerobot-auto-calibrate-feetech --port COM3 --save
  lerobot-auto-calibrate-feetech --port COM3 --unfold-angle 0
  lerobot-auto-calibrate-feetech --port COM3 --save --robot-id default
  lerobot-auto-calibrate-feetech --port COM3 --unfold-only   # Only debug arm unfolding (Stage 0 + Stage 2)
"""

import argparse
import contextlib
import sys
import time
from collections.abc import Callable

import draccus

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.auto_calibration import COMM_ERR
from lerobot.motors.feetech.calibration_defaults import (
    CALIBRATE_FIRST,
    CALIBRATE_REST,
    DEFAULT_ACCELERATION,
    DEFAULT_D_COEFFICIENT,
    DEFAULT_I_COEFFICIENT,
    DEFAULT_MAX_TORQUE,
    DEFAULT_P_COEFFICIENT,
    DEFAULT_POS_SPEED,
    DEFAULT_TIMEOUT,
    DEFAULT_TORQUE_LIMIT,
    DEFAULT_UNFOLD_ANGLE,
    DEFAULT_UNFOLD_TIMEOUT,
    DEFAULT_VELOCITY_LIMIT,
    FULL_TURN,
    HOMING_OFFSET_MAX_MAG,
    MOTOR_NAMES,
    SO_FOLLOWER_MOTORS,
    STS_HALF_TURN_RAW,
    UNFOLD_ORDER,
    motor_label,
)
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feetech servo auto-calibration (with unfolding): complete full workflow in one command."
    )
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port path, e.g. COM3 or /dev/ttyUSB0",
    )
    parser.add_argument(
        "--motor",
        type=str,
        choices=MOTOR_NAMES,
        default=None,
        help="Only test this motor (skip unfolding); if not specified, test all 6 motors sequentially",
    )

    cal = parser.add_argument_group("Calibration parameters")
    cal.add_argument(
        "--velocity-limit",
        type=int,
        default=DEFAULT_VELOCITY_LIMIT,
        help=f"Calibration limit-detection velocity (constant-speed mode Goal_Velocity), default {DEFAULT_VELOCITY_LIMIT}",
    )
    cal.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Calibration single-direction limit detection timeout (seconds), default {DEFAULT_TIMEOUT}",
    )
    unfold = parser.add_argument_group("Unfold parameters")
    unfold.add_argument(
        "--unfold-only",
        action="store_true",
        help="Only perform arm unfolding (Stage 0 init + Stage 2 unfold), no calibration, for debugging unfold logic",
    )
    unfold.add_argument(
        "--unfold-angle",
        type=float,
        default=DEFAULT_UNFOLD_ANGLE,
        help=f"Unfold angle (degrees), set to 0 to skip unfolding. Default {DEFAULT_UNFOLD_ANGLE}",
    )
    unfold.add_argument(
        "--unfold-timeout",
        type=float,
        default=DEFAULT_UNFOLD_TIMEOUT,
        help=f"Unfold single-motion wait timeout (seconds), default {DEFAULT_UNFOLD_TIMEOUT}",
    )
    out = parser.add_argument_group("Output (same path and format as manual calibration)")
    out.add_argument(
        "--save",
        action="store_true",
        help="Write calibration data to servo EEPROM and save to the same local path as manual calibration (draccus format)",
    )
    out.add_argument(
        "--robot-id",
        type=str,
        default="default",
        help="Robot id for saving, corresponds to path .../calibration/robots/<robot_type>/<robot_id>.json, must match config.id when starting the arm",
    )
    out.add_argument(
        "--robot-type",
        type=str,
        default="so_follower",
        choices=["so_follower", "so_leader"],
        help="Robot type for calibration file path: 'so_follower' (default) or 'so_leader'",
    )

    return parser.parse_args()


# ====================== Unfolding ======================


def _unfold_joints(
    bus: FeetechMotorsBus,
    unfold_angle: float,
    unfold_timeout: float,
    unfold_directions: dict[str, str | None] | None = None,
) -> None:
    """Unfold joints 2-4 to avoid mechanical interference during calibration. If unfold_directions is provided, record each joint's unfold direction."""
    print(f"\n{'=' * 20} Stage 2: Unfold joints 2-4 ({unfold_angle} deg) {'=' * 20}")
    for motor in UNFOLD_ORDER:
        direction, _ = bus.unfold_single_joint(motor, unfold_angle, unfold_timeout)
        if unfold_directions is not None and direction is not None:
            unfold_directions[motor] = direction
    print("\n  Unfolding complete, joints 2-4 are raised. Unfold direction for each joint:")
    if unfold_directions is not None:
        for motor in UNFOLD_ORDER:
            direction = unfold_directions.get(motor, "unknown")
            print(f"    {motor_label(motor)}: unfold direction = {direction}")


def _fold_arm(
    bus: FeetechMotorsBus,
    all_mins: dict[str, int],
    all_maxes: dict[str, int],
    all_unfold_directions: dict[str, str | None],
    *,
    motors: list[str] | None = None,
    unfold: bool = False,
    unfold_per_motor: dict[str, bool] | None = None,
) -> None:
    """Fold or fully unfold specified joints. Multiple servos move simultaneously.

    Fold (unfold=False): forward unfold -> fold target range_max; reverse -> range_min; gripper fixed at range_min.
    Unfold (unfold=True): target is opposite of fold, forward -> range_min, reverse -> range_max; gripper fixed at range_max.
    motors: list of servos to move; if None or empty, use default order (shoulder_lift->elbow_flex->wrist_flex->gripper).
    unfold_per_motor: optional, specify fold(False)/unfold(True) per joint; unlisted joints use unfold. If None, all use unfold.
    """
    default_order = ["shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]
    fold_order = motors if motors else default_order
    title = "Fold/unfold arm" if unfold_per_motor else ("Unfold" if unfold else "Fold") + " arm"
    print(f"\n{'=' * 20} {title} (simultaneous) {'=' * 20}")
    values: dict[str, tuple[int, int, int]] = {}

    for motor in fold_order:
        if motor not in all_mins or motor not in all_maxes:
            continue
        per_unfold = unfold_per_motor.get(motor, unfold) if unfold_per_motor is not None else unfold
        direction = all_unfold_directions.get(motor)
        # Compute fold end and unfold end, then select based on per_unfold
        if motor == "gripper":
            fold_end = all_mins[motor]
            unfold_end = all_maxes[motor]
        else:
            fold_end = all_maxes[motor] if direction == "reverse" else all_mins[motor]
            unfold_end = all_mins[motor] if direction == "reverse" else all_maxes[motor]
        target = unfold_end if per_unfold else fold_end
        label = "range_max" if target == all_maxes[motor] else "range_min"
        if motor == "gripper":
            label += "(gripper)" if per_unfold else "(gripper forward)"
        action_m = "unfold" if per_unfold else "fold"
        values[motor] = (target, DEFAULT_POS_SPEED, DEFAULT_ACCELERATION)
        bus.write("Operating_Mode", motor, 0)  # servo mode
        try:
            pos = bus.read("Present_Position", motor, normalize=False)
            print(f"  {motor_label(motor)} current pos={pos}, {action_m} to {label}={target}.")
        except COMM_ERR:
            print(f"  {motor_label(motor)} failed to read current pos, {action_m} to {label}={target}.")
    if not values:
        action = "unfold" if unfold else "fold"
        print(f"  No valid motors, skipping {action}.\n")
        return
    bus.sync_write_pos_ex(values)
    time.sleep(0.3)
    # Poll until all motors stop
    timeout_s = 10.0
    poll_s = 0.05
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            if all(bus.read("Moving", m, normalize=False) == 0 for m in values):
                break
        except COMM_ERR:
            pass
        time.sleep(poll_s)
    done_label = "fold/unfold" if unfold_per_motor else ("unfold" if unfold else "fold")
    for m in values:
        try:
            pos = bus.read("Present_Position", m, normalize=False)
            print(f"  {motor_label(m)} after {done_label}: end pos={pos}, reached target")
        except COMM_ERR:
            print(f"  {motor_label(m)} after {done_label}: failed to read pos, reached target")
    print(f"  {done_label} complete.\n")


def _move_arm_by_angle(
    bus: FeetechMotorsBus,
    all_unfold_directions: dict[str, str | None],
    angle_deg: float,
    *,
    fold: bool = False,
    motors: list[str] | None = None,
    all_mins: dict[str, int] | None = None,
    all_maxes: dict[str, int] | None = None,
) -> None:
    """Move specified degrees from current position in unfold or fold direction. Does not detect direction; relies on all_unfold_directions.

    Direction consistent with _fold_arm: forward unfold -> position increase is unfold, decrease is fold; reverse unfold -> position decrease is unfold, increase is fold.
    fold: False=unfold direction, True=fold direction.
    motors: list of servos to move, None or empty uses default order (shoulder_lift->elbow_flex->wrist_flex).
    all_mins/all_maxes: optional, if provided, clamp target position to limits.
    """
    default_order = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    move_order = motors if motors else default_order
    angle_steps = int(angle_deg / 360.0 * FULL_TURN)
    direction_label = "fold" if fold else "unfold"
    print(f"\n{'=' * 20} Relative {direction_label} {angle_deg:.1f} deg from current pos {'=' * 20}")
    for motor in move_order:
        if (
            all_mins is not None
            and all_maxes is not None
            and (motor not in all_mins or motor not in all_maxes)
        ):
            continue
        try:
            present = bus.read("Present_Position", motor, normalize=False)
        except COMM_ERR:
            print(f"  Warning: {motor_label(motor)} failed to read current position, skipping")
            continue
        direction = all_unfold_directions.get(motor)
        # Consistent with _fold_arm: forward -> unfold increases position, fold decreases; reverse -> opposite
        if fold:
            target = present - angle_steps if direction == "forward" else present + angle_steps
        else:
            target = present + angle_steps if direction == "forward" else present - angle_steps
        if all_mins is not None and all_maxes is not None and motor in all_mins and motor in all_maxes:
            target = max(all_mins[motor], min(all_maxes[motor], target))
        print(f"  {motor_label(motor)} {direction_label} {angle_deg:.1f} deg: pos {present} -> {target}")
        ok = bus.write_pos_ex_and_wait(
            motor,
            target,
            DEFAULT_POS_SPEED,
            DEFAULT_ACCELERATION,
            timeout_s=DEFAULT_UNFOLD_TIMEOUT,
            poll_interval_s=0.05,
        )
        if not ok:
            print(f"  Warning: {motor_label(motor)} motion timed out, keeping current position")
        else:
            print(f"  {motor_label(motor)} reached target")
    print(f"  {direction_label} complete.\n")


# ====================== Calibration ======================


def _record_reference_position(
    bus: FeetechMotorsBus,
    motor_name: str,
    out: dict[str, int],
) -> None:
    """Read the motor's current reference position (Present_Position + Homing_Offset) % FULL_TURN and store in out[motor_name]. On read failure, out is not modified."""
    try:
        pr = bus.read("Present_Position", motor_name, normalize=False)
        ho = bus.read("Homing_Offset", motor_name, normalize=False)
        out[motor_name] = (pr + ho) % FULL_TURN
    except COMM_ERR:
        pass


def _calibrate_motors(
    bus: FeetechMotorsBus,
    motor_names: list[str],
    *,
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    ccw_first: bool = False,
    unfold_directions: dict[str, str | None] | None = None,
    reference_positions: dict[str, int] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Calibrate a group of motors (run measure_ranges_of_motion_multi then write back and return to center). Returns {motor_name: (range_min, range_max, mid_raw)}.
    If unfold_directions is provided and motors 2 and 3 are calibrated together: both move simultaneously, motor 2 fully folds, motor 3 fully unfolds; otherwise return to center as normal.
    If reference_positions is provided: use that reference position to select arc for the corresponding motor, skipping limit retreat and re-read."""
    if not motor_names:
        return {}
    raw_results = bus.measure_ranges_of_motion_multi(
        motor_names,
        velocity_limit=velocity_limit,
        timeout_s=timeout_s,
        ccw_first=ccw_first,
        reference_positions=reference_positions,
    )
    print("Preparing to write registers")
    result: dict[str, tuple[int, int, int]] = {}
    for m in motor_names:
        rmin, rmax, mid_raw, _raw_min_meas, _raw_max_meas, homing_offset = raw_results[m]
        print(
            f"  {motor_label(m)}: after offset range_min={rmin}, range_max={rmax}, mid={mid_raw}, Homing_Offset register={homing_offset}"
        )
        time.sleep(0.05)
        try:
            ho_before = bus.read("Homing_Offset", m, normalize=False)
            min_before = bus.read("Min_Position_Limit", m, normalize=False)
            max_before = bus.read("Max_Position_Limit", m, normalize=False)
            print(
                f"  {motor_label(m)} before write: Min_Position_Limit={min_before}, Max_Position_Limit={max_before}, Homing_Offset={ho_before}"
            )
        except COMM_ERR:
            print(f"  {motor_label(m)} before write: failed to read registers")
        bus.safe_write("Homing_Offset", m, homing_offset, normalize=False)
        bus.safe_write_position_limits(m, rmin, rmax)
        time.sleep(0.1)
        try:
            ho_after = bus.read("Homing_Offset", m, normalize=False)
            min_after = bus.read("Min_Position_Limit", m, normalize=False)
            max_after = bus.read("Max_Position_Limit", m, normalize=False)
            print(
                f"  {motor_label(m)} after write: Min_Position_Limit={min_after}, Max_Position_Limit={max_after}, Homing_Offset={ho_after}"
            )
        except COMM_ERR:
            print(f"  {motor_label(m)} after write: failed to read registers")
        time.sleep(0.1)
        do_2_3_together = (
            unfold_directions is not None and "shoulder_lift" in motor_names and "elbow_flex" in motor_names
        )
        if m == "wrist_roll":
            pass
        elif do_2_3_together and m in ("shoulder_lift", "elbow_flex"):
            # Motors 2 and 3 keep torque locked
            pass
        else:
            bus.go_to_mid(m)
        result[m] = (rmin, rmax, mid_raw)

    return result


# ====================== Connection and initialization (shared) ======================


def _connect_and_clear(port: str) -> FeetechMotorsBus:
    """Create bus, clear residual Overload, then formally connect. Raises exception on failure."""
    bus = FeetechMotorsBus(port=port, motors=SO_FOLLOWER_MOTORS.copy())
    bus.connect(handshake=False)
    print("Clearing residual servo state...")
    all_zero = dict.fromkeys(MOTOR_NAMES, 0)
    for _ in range(3):
        with contextlib.suppress(COMM_ERR):
            bus.sync_write("Goal_Velocity", all_zero)
        with contextlib.suppress(COMM_ERR):
            bus.sync_write("Torque_Enable", all_zero)
        time.sleep(0.2)
    bus.disconnect(disable_torque=False)
    time.sleep(0.2)
    bus.connect()
    print("All servos ready.")
    return bus


def _run_with_bus(
    port: str,
    interactive: bool,
    body: Callable[[FeetechMotorsBus], None],
) -> int:
    """Connect bus then execute body(bus), with unified handling of connection failure, KeyboardInterrupt, Exception and disconnect. Returns 0 success, 1 error, 130 user interrupt."""
    try:
        bus = _connect_and_clear(port)
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1
    try:
        body(bus)
    except KeyboardInterrupt:
        print("\nUser interrupted, disabling all servos...")
        bus.safe_disable_all()
        return 130
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        bus.safe_disable_all()
        if interactive:
            with contextlib.suppress(EOFError):
                input("Press Enter to exit...")
        return 1
    finally:
        bus.disconnect()
    return 0


# Stage 0 initialization: write->read->compare per register, table-driven config; special items (limits, Torque_Enable) handled separately
INIT_CHECKS = [
    ("Lock", 1),
    ("Return_Delay_Time", 0),
    ("Operating_Mode", 0),
    ("Max_Torque_Limit", DEFAULT_MAX_TORQUE),
    ("Torque_Limit", DEFAULT_TORQUE_LIMIT),
    ("Acceleration", DEFAULT_ACCELERATION),
    ("P_Coefficient", DEFAULT_P_COEFFICIENT),
    ("I_Coefficient", DEFAULT_I_COEFFICIENT),
    ("D_Coefficient", DEFAULT_D_COEFFICIENT),
    ("Homing_Offset", 0),
]


def _run_init(bus: FeetechMotorsBus, *, interactive: bool = True) -> None:
    """Stage 0: Lock=1, PID, limits, Homing_Offset, enable torque. If parameter error and interactive, wait for Enter."""
    print(f"\n{'=' * 20} Stage 0: Initialize {'=' * 20}")
    for m in MOTOR_NAMES:
        print(f"Configuring servo: {motor_label(m)}")
        try:
            bus.write("Torque_Enable", m, 0)
            time.sleep(0.05)
        except COMM_ERR:
            pass
        param_set_ok = True
        try:
            for reg, expected in INIT_CHECKS:
                bus.write(reg, m, expected, normalize=(reg != "Homing_Offset"))
                time.sleep(0.01)
                got = bus.read(reg, m, normalize=False)
                if got != expected:
                    print(f"  [Warning] {reg} set failed on {m}: expected={expected}, got={got}")
                    param_set_ok = False
            # Position limits: write/read/compare separately
            bus.write_position_limits(m, 0, 4095)
            time.sleep(0.05)
            limits = bus.read_position_limits(m)
            if limits != (0, 4095):
                print(f"  [Warning] Position_Limits set failed on {m}: expected=(0, 4095), got={limits}")
                param_set_ok = False
            time.sleep(0.2)
            # Finally enable torque
            bus.write("Torque_Enable", m, 1)
            time.sleep(0.05)
            te_read = bus.read("Torque_Enable", m, normalize=False)
            if te_read != 1:
                print(f"  [Warning] Torque_Enable failed on {m}: expected=1, got={te_read}")
                param_set_ok = False
            time.sleep(0.1)
        except Exception as e:
            print(f"  [Exception] Error setting parameters on {m}: {e}")
            param_set_ok = False
        if not param_set_ok and interactive:
            with contextlib.suppress(Exception):
                input(
                    "  [Warning] Parameter set/verify failed, check wiring and power, press Enter to force continue..."
                )
    print(
        f"Initialized and torque enabled (P={DEFAULT_P_COEFFICIENT}, "
        f"Acc={DEFAULT_ACCELERATION}, Torque={DEFAULT_TORQUE_LIMIT})."
    )


# ====================== Public entry points (full calibration / unfold only / single motor) ======================


def _apply_calibration_results(
    results: dict[str, tuple[int, int, int]],
    all_mins: dict[str, int],
    all_maxes: dict[str, int],
    all_mids: dict[str, int],
    motor_list: list[str],
) -> None:
    """Write _calibrate_motors results into all_mins / all_maxes / all_mids."""
    for m in motor_list:
        all_mins[m], all_maxes[m], all_mids[m] = results[m]


def run_full_calibration(
    port: str,
    *,
    save: bool = False,
    robot_id: str = "default",
    robot_type: str = "so_follower",
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    unfold_timeout_s: float = DEFAULT_UNFOLD_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Full calibration workflow: initialize -> servos 2-6 (with arm raise for clearance) -> servo 1 shoulder_pan calibrated last -> fold.
    If save is True: write to servo EEPROM and save to the same path and format as manual calibration (draccus, loaded at arm startup).

    Called by CLI or teleoperation programs. Returns 0 success, 1 error, 130 user interrupt.
    """

    def body(bus: FeetechMotorsBus) -> None:
        all_mins: dict[str, int] = {}
        all_maxes: dict[str, int] = {}
        all_mids: dict[str, int] = {}
        all_unfold_directions: dict[str, str | None] = {}
        all_reference_positions: dict[str, int] = {}
        _run_init(bus, interactive=interactive)
        # Raise motor 4 by 80 degrees
        direction, _ = bus.unfold_single_joint("wrist_flex", 80, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["wrist_flex"] = direction
        time.sleep(0.1)
        # Raise motors 2 and 3 and record reference positions (Present_Position + Homing_Offset, used for arc selection during calibration)
        direction, _ = bus.unfold_single_joint("shoulder_lift", 15, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["shoulder_lift"] = direction
        _record_reference_position(bus, "shoulder_lift", all_reference_positions)
        direction, _ = bus.unfold_single_joint("elbow_flex", 30, unfold_timeout_s)
        if direction is not None:
            all_unfold_directions["elbow_flex"] = direction
        _record_reference_position(bus, "elbow_flex", all_reference_positions)
        time.sleep(0.1)
        # Fold: retract shoulder_lift and elbow_flex
        for m in ["shoulder_lift", "elbow_flex"]:
            bus.go_to_mid(m)
            time.sleep(0.1)
        # Calibrate motors 2 and 3 using multi-motor calibration, initial rotation direction is opposite to their raise direction
        # forward raise -> CCW first; reverse raise -> CW first; default CCW first if not recorded
        ccw_first_2_3 = {
            "shoulder_lift": all_unfold_directions.get("shoulder_lift") != "reverse",
            "elbow_flex": all_unfold_directions.get("elbow_flex") != "reverse",
        }
        print(f"\n{'=' * 20} Calibrate motors 2 and 3 (multi-motor, opposite to raise direction) {'=' * 20}")
        results_2_3 = _calibrate_motors(
            bus,
            ["shoulder_lift", "elbow_flex"],
            velocity_limit=velocity_limit,
            timeout_s=timeout_s,
            ccw_first=ccw_first_2_3,
            unfold_directions=all_unfold_directions,
            reference_positions=all_reference_positions,
        )
        _apply_calibration_results(
            results_2_3, all_mins, all_maxes, all_mids, ["shoulder_lift", "elbow_flex"]
        )
        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions, motors=["shoulder_lift", "elbow_flex"])

        time.sleep(0.1)
        # Stage 3: Calibrate remaining motors 4, 5, 6 (multi-motor simultaneous, with arm raise for clearance)
        print(f"\n{'=' * 20} Stage 3: Calibrate motors 4-6 (multi-motor simultaneous) {'=' * 20}")
        _move_arm_by_angle(
            bus,
            all_unfold_directions,
            80,
            fold=False,
            motors=["elbow_flex"],
            all_mins=all_mins,
            all_maxes=all_maxes,
        )
        calibrate_rest_remaining = ["wrist_roll", "gripper", "wrist_flex"]
        results_rest = _calibrate_motors(
            bus,
            calibrate_rest_remaining,
            velocity_limit=velocity_limit,
            timeout_s=timeout_s,
            reference_positions=all_reference_positions,
        )
        _apply_calibration_results(results_rest, all_mins, all_maxes, all_mids, calibrate_rest_remaining)
        time.sleep(0.1)
        # Fold motor 3, fully unfold motor 4 (executed simultaneously in one call)
        _fold_arm(
            bus,
            all_mins,
            all_maxes,
            all_unfold_directions,
            motors=["elbow_flex", "wrist_flex", "gripper"],
            unfold_per_motor={"elbow_flex": False, "wrist_flex": True, "gripper": False},
        )
        # Stage 4: Finally calibrate motor 1 shoulder_pan
        print(
            f"\n{'=' * 20} Stage 4: Calibrate {motor_label('shoulder_pan')} (motor 1) and return to center {'=' * 20}"
        )
        results_pan = _calibrate_motors(
            bus, ["shoulder_pan"], velocity_limit=velocity_limit, timeout_s=timeout_s
        )
        _apply_calibration_results(results_pan, all_mins, all_maxes, all_mids, ["shoulder_pan"])
        time.sleep(0.1)
        motors_calibrated = CALIBRATE_REST + CALIBRATE_FIRST
        print(f"\n{'=' * 20} Calibration results {'=' * 20}")
        for name in motors_calibrated:
            offset = all_mids[name] - STS_HALF_TURN_RAW
            print(
                f"  {motor_label(name)}: min={all_mins[name]}, max={all_maxes[name]}, "
                f"mid={all_mids[name]}, offset={offset}"
            )

        _fold_arm(bus, all_mins, all_maxes, all_unfold_directions)
        # Before persisting: unlock EEPROM (Lock=0) and restore all servos to servo mode (Operating_Mode=0)
        for name in bus.motors:
            bus.write("Lock", name, 0)
            time.sleep(0.01)
            bus.write("Operating_Mode", name, 0)
            time.sleep(0.01)
        time.sleep(1)
        if interactive:
            bus.safe_disable_all()
            print("\nCalibration complete.")

        if save:
            print(f"\n{'=' * 20} Persisting (same method as manual calibration) {'=' * 20}")
            bus.safe_disable_all()
            cal = {}
            for name in motors_calibrated:
                m = SO_FOLLOWER_MOTORS[name]
                offset = all_mids[name] - STS_HALF_TURN_RAW
                offset = max(-HOMING_OFFSET_MAX_MAG, min(HOMING_OFFSET_MAX_MAG, offset))
                cal[name] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=offset,
                    range_min=all_mins[name],
                    range_max=all_maxes[name],
                )
            bus.write_calibration(cal, cache=True)
            print("Calibration written to servo EEPROM.")
            # Same path and format as manual calibration, loaded at arm startup
            calibration_fpath = HF_LEROBOT_CALIBRATION / "robots" / robot_type / f"{robot_id}.json"
            calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(calibration_fpath, "w") as f, draccus.config_type("json"):
                draccus.dump(cal, f, indent=4)
            print(f"Calibration saved to: {calibration_fpath}")
        print("Disabling all servos...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


def unfold_joints(
    port: str,
    angle_deg: float,
    *,
    timeout_s: float = DEFAULT_UNFOLD_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Only perform Stage 0 init + unfold joints 2-4 to specified angle. For debugging unfold. Returns 0/1/130."""

    def body(bus: FeetechMotorsBus) -> None:
        _run_init(bus, interactive=interactive)
        all_unfold_directions: dict[str, str | None] = {}
        if angle_deg > 0:
            _unfold_joints(bus, angle_deg, timeout_s, all_unfold_directions)
            print("  Unfolding complete, unfold directions:")
            for motor, direction in all_unfold_directions.items():
                print(f"    {motor_label(motor)}: {direction}")
            if interactive:
                input("  Press Enter to disable torque and exit...")
        else:
            print("  Unfold angle is 0, unfolding skipped.")
            if interactive:
                input("  Press Enter to disable torque and exit...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


def calibrate_single_motor(
    port: str,
    motor_name: str,
    *,
    velocity_limit: int = DEFAULT_VELOCITY_LIMIT,
    timeout_s: float = DEFAULT_TIMEOUT,
    interactive: bool = True,
) -> int:
    """Only perform Stage 0 + calibrate specified motor, no folding, no saving. For testing. Returns 0/1/130."""

    def body(bus: FeetechMotorsBus) -> None:
        _run_init(bus, interactive=interactive)
        print(f"\n{'=' * 20} Calibrate {motor_label(motor_name)} {'=' * 20}")
        _calibrate_motors(bus, [motor_name], velocity_limit=velocity_limit, timeout_s=timeout_s)
        time.sleep(0.1)
        if interactive:
            input("  Calibration complete, press Enter to disable torque and exit...")
        bus.safe_disable_all()

    return _run_with_bus(port, interactive, body)


# ====================== CLI entry point ======================


def main() -> int:
    """CLI: dispatch to full calibration, unfold only, or single motor calibration based on arguments."""
    args = parse_args()
    print(f"Serial port: {args.port}")
    if getattr(args, "unfold_only", False):
        print("Unfold only (--unfold-only): Stage 0 init + Stage 2 unfold, no calibration")
        print(f"Unfold angle: {args.unfold_angle} deg")
        return unfold_joints(
            args.port,
            args.unfold_angle,
            timeout_s=args.unfold_timeout,
            interactive=True,
        )
    if args.motor is not None:
        print(f"Single motor mode: {args.motor}")
        return calibrate_single_motor(
            args.port,
            args.motor,
            velocity_limit=args.velocity_limit,
            timeout_s=args.timeout,
            interactive=True,
        )
    print(f"Full calibration: {CALIBRATE_FIRST + CALIBRATE_REST}")
    return run_full_calibration(
        args.port,
        save=args.save,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        velocity_limit=args.velocity_limit,
        timeout_s=args.timeout,
        unfold_timeout_s=args.unfold_timeout,
        interactive=True,
    )


if __name__ == "__main__":
    sys.exit(main())
