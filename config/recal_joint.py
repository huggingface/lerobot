#!/usr/bin/env python3
"""Recalibrate ONE joint after a servo swap. Every other joint keeps its calibration.

    ./config/recal-joint.sh right_arm_wrist_roll

lerobot-calibrate redoes the whole robot - both arms, the head, and two
rounds of moving every joint through its range - and any small difference
in where you hold the other thirteen joints changes how they track their
leaders. A replaced servo needs only its own two numbers redone:

  homing offset   where "the middle" is. A new servo's encoder magnet is not
                  where the old one's was, so the old offset is meaningless.
  range           the raw positions at the ends of travel.

Both are written to that servo and to that joint's entry in the calibration
file. The file is backed up first, next to itself, with a timestamp.

THE POSE MATTERS. "Middle" has to be the same physical pose the joint was in
when the arm was first calibrated, or the follower will be offset from its
leader by however far off it is. For a wrist roll, compare with the other
arm: hold the gripper in the same orientation the other arm's gripper has in
its middle pose (mirrored). If teleop afterwards shows the wrist turned
relative to the leader, run this again and hold it that much the other way.
"""

import argparse
import json
import shutil
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

PORTS = {
    1: "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00",
    2: "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00",
}
JOINTS = {
    "left_arm_shoulder_pan": (1, 1), "left_arm_shoulder_lift": (1, 2), "left_arm_elbow_flex": (1, 3),
    "left_arm_wrist_flex": (1, 4), "left_arm_wrist_roll": (1, 5), "left_arm_gripper": (1, 6),
    "head_motor_1": (1, 7), "head_motor_2": (1, 8),
    "right_arm_shoulder_pan": (2, 1), "right_arm_shoulder_lift": (2, 2), "right_arm_elbow_flex": (2, 3),
    "right_arm_wrist_flex": (2, 4), "right_arm_wrist_roll": (2, 5), "right_arm_gripper": (2, 6),
}


def ask(prompt: str) -> bool:
    try:
        input(prompt)
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joint", choices=sorted(JOINTS))
    ap.add_argument("--id", dest="robot_id", default="xlerobot", help="robot id (calibration file name)")
    args = ap.parse_args()

    bus_n, motor_id = JOINTS[args.joint]
    fpath = HF_LEROBOT_CALIBRATION / ROBOTS / "xlerobot" / f"{args.robot_id}.json"
    if not fpath.is_file():
        print(f"error: no calibration file at {fpath}", file=sys.stderr)
        return 1
    cal = json.loads(fpath.read_text())
    if args.joint not in cal:
        print(f"error: {fpath} has no entry for {args.joint}; it has {sorted(cal)}", file=sys.stderr)
        return 1
    old = dict(cal[args.joint])

    print(f"\njoint  : {args.joint}  (bus {bus_n}, id {motor_id})")
    print(f"file   : {fpath}")
    print(f"current: homing_offset {old.get('homing_offset')}, "
          f"range {old.get('range_min')}..{old.get('range_max')}")

    name = args.joint
    norm = MotorNormMode.RANGE_0_100 if name.endswith("gripper") else MotorNormMode.RANGE_M100_100
    bus = FeetechMotorsBus(port=PORTS[bus_n], motors={name: Motor(motor_id, "sts3215", norm)})
    bus.connect(handshake=False)
    try:
        model = bus.ping(motor_id, num_retry=3)
        if model is None:
            print(f"\nid {motor_id} does not answer on bus {bus_n}. Run ./config/scan-motors.sh.")
            return 1

        bus.write("Torque_Enable", name, 0, num_retry=3)
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value, num_retry=3)

        print("\nStep 1 of 2 - the middle.")
        print("  The joint is limp now. Hold it in its MIDDLE pose: the same pose it")
        print("  was in when the arm was first calibrated. For a wrist roll, match the")
        print("  other arm's gripper orientation in its middle pose (mirrored).")
        if not ask("  ENTER while holding it there, Ctrl-C to stop (nothing written yet): "):
            return 1
        offsets = bus.set_half_turn_homings([name])
        new_offset = offsets[name]
        print(f"  homing offset written: {new_offset}")

        print("\nStep 2 of 2 - the range.")
        print("  Move the joint slowly to one end of its travel, then to the other,")
        print("  then back to the middle. Press ENTER when done.")
        time.sleep(0.5)
        mins, maxes = bus.record_ranges_of_motion([name])
        new = {
            "id": motor_id,
            "drive_mode": old.get("drive_mode", 0),
            "homing_offset": int(new_offset),
            "range_min": int(mins[name]),
            "range_max": int(maxes[name]),
        }
        bus.write_calibration({name: MotorCalibration(**new)})
    finally:
        bus.disconnect(disable_torque=True)

    backup = fpath.with_name(f"{fpath.stem}.before-{args.joint}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copy2(fpath, backup)
    cal[args.joint] = new
    fpath.write_text(json.dumps(cal, indent=4))

    print("\n  " + "-" * 50)
    print(f"  {'':<14}{'before':>12}{'after':>12}")
    for k in ("homing_offset", "range_min", "range_max"):
        print(f"  {k:<14}{str(old.get(k)):>12}{new[k]:>12}")
    print(f"\n  saved   {fpath}")
    print(f"  backup  {backup}")
    span = new["range_max"] - new["range_min"]
    if span < 1000:
        print(f"\n  WARNING: only {span} steps of travel recorded - did the joint go end to end?")
    print("\n  Next: BASE=none ./config/cart-selftest.sh --arms, then teleop and check the")
    print("  joint tracks its leader. Offset in teleop = redo step 1 with a corrected pose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
