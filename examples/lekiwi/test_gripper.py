#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Automated validation for the LeKiwi gripper (arm_gripper, id 6).

Diagnoses the common "gripper doesn't move when the leader sends commands" problem by separating two
independent questions:

  1. Is the servo mechanically responsive?  -> drive it with RAW goal positions (bypasses calibration)
     around its current position and measure the travel actually achieved.
  2. Is the calibration usable?             -> the leader/client send normalized 0..100 commands, which
     are mapped into [range_min, range_max]. If that span is tiny, every command collapses onto a few
     raw ticks and the gripper can't move even though the servo is fine.

Run against a robot that is NOT currently owned by a running host (the host holds the serial port):

    uv run python -m examples.lekiwi.test_gripper --id my_awesome_kiwi

Exit codes: 0 = gripper healthy, 1 = calibration bad (recalibrate), 2 = servo/comms problem.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

GRIPPER = "arm_gripper"
GRIPPER_ID = 6

# A gripper's usable travel is a large fraction of the 0..4095 encoder range. Anything smaller than this
# span in the calibration means normalized commands collapse to a few ticks -> gripper looks "dead".
MIN_HEALTHY_CALIB_SPAN = 200
# The servo is considered mechanically responsive if a raw sweep moves it at least this many ticks.
MIN_MECHANICAL_TRAVEL = 80


def load_gripper_calibration(robot_id: str) -> MotorCalibration | None:
    """Read the gripper entry from the on-disk calibration file for this robot id, if present."""
    calib_path = (
        Path.home()
        / ".cache/huggingface/lerobot/calibration/robots/lekiwi"
        / f"{robot_id}.json"
    )
    if not calib_path.is_file():
        print(f"  (no calibration file at {calib_path})")
        return None
    data = json.loads(calib_path.read_text())
    g = data.get(GRIPPER)
    if not g:
        return None
    return MotorCalibration(
        id=g["id"],
        drive_mode=g["drive_mode"],
        homing_offset=g["homing_offset"],
        range_min=g["range_min"],
        range_max=g["range_max"],
    )


def sweep_raw(bus: FeetechMotorsBus, center: int, delta: int, step: int, settle: float) -> list[int]:
    """Ramp the gripper to (center - delta) then (center + delta) then back, in RAW ticks.

    Ramps in small steps and aborts a direction early if the servo stops tracking (i.e. it hit a hard
    stop), so we never stall torque into an end stop. Returns every observed raw position.
    """
    observed = [center]

    def ramp_to(target: int) -> None:
        target = max(0, min(4095, target))
        cur = bus.read("Present_Position", GRIPPER, normalize=False, num_retry=3)
        direction = 1 if target >= cur else -1
        for goal in range(cur, target + direction, direction * step):
            goal = max(0, min(4095, goal))
            bus.write("Goal_Position", GRIPPER, goal, normalize=False, num_retry=3)
            time.sleep(settle)
            actual = bus.read("Present_Position", GRIPPER, normalize=False, num_retry=3)
            observed.append(actual)
            # If we command a move but the servo isn't following, treat it as an end stop and back off.
            if abs(goal - actual) > max(2 * step, 60):
                print(f"    stopped tracking near raw={actual} (commanded {goal}); assuming end stop")
                break

    ramp_to(center - delta)
    ramp_to(center + delta)
    ramp_to(center)  # return home
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default="my_awesome_kiwi", help="Robot id (for calibration lookup).")
    parser.add_argument("--port", default=None, help="Serial port (default: LeKiwiConfig default).")
    parser.add_argument("--delta", type=int, default=150, help="Raw ticks to sweep each side of center.")
    parser.add_argument("--step", type=int, default=30, help="Raw ticks per ramp step.")
    parser.add_argument("--settle", type=float, default=0.15, help="Seconds to wait after each step.")
    parser.add_argument("--no-move", action="store_true", help="Only inspect calibration; do not move.")
    parser.add_argument(
        "--test-normalized",
        action="store_true",
        help="Also send normalized 0/100 commands (the exact leader path) and measure raw travel.",
    )
    args = parser.parse_args()

    port = args.port or LeKiwiConfig().port

    # ---- 1. Calibration inspection (no hardware needed) -------------------------------------------
    print("== Gripper calibration ==")
    calib = load_gripper_calibration(args.id)
    calib_ok = False
    if calib is None:
        print("  No gripper calibration found -> normalized commands will fail. Recalibrate.")
    else:
        span = calib.range_max - calib.range_min
        print(f"  id={calib.id} homing_offset={calib.homing_offset} "
              f"range_min={calib.range_min} range_max={calib.range_max} span={span}")
        calib_ok = span >= MIN_HEALTHY_CALIB_SPAN
        if calib_ok:
            print(f"  OK: span {span} >= {MIN_HEALTHY_CALIB_SPAN}.")
        else:
            print(f"  BAD: span {span} < {MIN_HEALTHY_CALIB_SPAN}. Normalized 0..100 commands collapse "
                  f"onto raw [{calib.range_min}, {calib.range_max}] -> gripper can't move. Recalibrate.")

    if args.no_move:
        return 0 if calib_ok else 1

    # ---- 2. Live servo check ----------------------------------------------------------------------
    calib_for_bus = {GRIPPER: calib} if calib is not None else {}
    bus = FeetechMotorsBus(
        port=port,
        motors={GRIPPER: Motor(GRIPPER_ID, "sts3215", MotorNormMode.RANGE_0_100)},
        calibration=calib_for_bus,
    )

    print(f"\n== Connecting to gripper (id {GRIPPER_ID}) on {port} ==")
    try:
        bus.connect()
    except Exception as e:
        print(f"  FAILED to connect / handshake with the gripper servo: {e}")
        print("  -> Check power and the daisy-chain cable, and that no host owns the serial port.")
        return 2

    center = None
    try:
        # Position mode + torque on (mode change requires torque off first).
        bus.disable_torque(GRIPPER)
        bus.write("Operating_Mode", GRIPPER, OperatingMode.POSITION.value)
        bus.enable_torque(GRIPPER)

        center = bus.read("Present_Position", GRIPPER, normalize=False, num_retry=3)
        print(f"  Current raw position: {center}")

        print(f"\n== Mechanical sweep (RAW +/-{args.delta} ticks, bypasses calibration) ==")
        observed = sweep_raw(bus, center, args.delta, args.step, args.settle)
        travel = max(observed) - min(observed)
        mech_ok = travel >= MIN_MECHANICAL_TRAVEL
        print(f"  Observed raw range: [{min(observed)}, {max(observed)}] -> travel {travel} ticks")
        print(f"  {'OK' if mech_ok else 'FAIL'}: servo "
              f"{'moves' if mech_ok else 'did NOT move'} (threshold {MIN_MECHANICAL_TRAVEL}).")

        norm_travel = None
        if args.test_normalized and calib is not None:
            print("\n== Normalized command path (what the leader sends: 0..100) ==")
            raws = []
            for cmd in (0.0, 100.0, 50.0):
                bus.write("Goal_Position", GRIPPER, cmd, normalize=True, num_retry=3)
                time.sleep(0.6)
                raws.append(bus.read("Present_Position", GRIPPER, normalize=False, num_retry=3))
                print(f"    cmd={cmd:5.1f} -> raw={raws[-1]}")
            norm_travel = max(raws) - min(raws)
            print(f"  Normalized-command raw travel: {norm_travel} ticks "
                  f"(tiny => confirms calibration is the bottleneck, not the servo).")

    finally:
        # Return home and release torque so nothing is left stalled.
        try:
            if center is not None:
                bus.write("Goal_Position", GRIPPER, center, normalize=False, num_retry=3)
                time.sleep(0.3)
            bus.disable_torque(GRIPPER)
        except Exception as e:
            print(f"  (cleanup warning: {e})")
        bus.disconnect()

    # ---- Verdict ----------------------------------------------------------------------------------
    print("\n== Verdict ==")
    if mech_ok and calib_ok:
        print("  Gripper healthy: servo moves and calibration span is usable.")
        return 0
    if mech_ok and not calib_ok:
        print("  Servo is FINE, but calibration is BAD -> this is why the leader can't move the gripper.")
        print("  Fix: recalibrate the gripper's range of motion (e.g. `lerobot-calibrate` for this robot,")
        print("  or rerun the LeKiwi host once and choose 'c' to recalibrate), then retest.")
        return 1
    print("  Servo did not move under direct RAW commands -> hardware issue (power/cable/servo), not calibration.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
