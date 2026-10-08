#!/usr/bin/env python
"""Probe both Feetech buses and say which xlerobot motors actually answer.

Run this BEFORE calibrating anything, and again after moving hardware to a
new machine. It is the only way to tell "the head servos are not wired yet"
apart from "the head servos are wired and one is dead" - both of which
surface later as the same unhelpful "no status packet" during connect.

    uv run --project ../XLeRobot python config/scan_motors.py
    uv run --project ../XLeRobot python config/scan_motors.py --port1 /dev/ttyACM0

Nothing is written. The scan only pings, and it sweeps every supported baud
rate, so a motor left on the wrong baud rate shows up here rather than
vanishing.
"""

from __future__ import annotations

import argparse
import sys

from lerobot.motors.feetech import FeetechMotorsBus

# What XLerobot expects, from xlerobot.py. Keep in step with it.
BUS1 = {
    1: "left_arm_shoulder_pan",
    2: "left_arm_shoulder_lift",
    3: "left_arm_elbow_flex",
    4: "left_arm_wrist_flex",
    5: "left_arm_wrist_roll",
    6: "left_arm_gripper",
    7: "head_motor_1",
    8: "head_motor_2",
}
BUS2 = {
    1: "right_arm_shoulder_pan",
    2: "right_arm_shoulder_lift",
    3: "right_arm_elbow_flex",
    4: "right_arm_wrist_flex",
    5: "right_arm_wrist_roll",
    6: "right_arm_gripper",
    7: "base_left_wheel",
    8: "base_back_wheel",
    9: "base_right_wheel",
}

DEFAULT_PORT1 = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00"
DEFAULT_PORT2 = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00"

# What lerobot configures motors to. A motor answering anywhere else needs
# setting up, not calibrating. Read from the class rather than hardcoded so
# an upstream change cannot make this quietly wrong.
EXPECTED_BAUD = FeetechMotorsBus.default_baudrate


def scan(port: str, expected: dict[int, str], label: str) -> tuple[set[int], bool]:
    print(f"\n{'=' * 68}\n{label}\n  {port}\n{'=' * 68}")
    try:
        found = FeetechMotorsBus.scan_port(port)
    except Exception as e:
        print(f"  ERROR: could not open the port - {e}")
        print("\n  Present now:")
        import glob

        for p in sorted(glob.glob("/dev/serial/by-id/*")) or ["    (none)"]:
            print(f"    {p}")
        return set(), False

    if not found:
        print("  Nothing answered at any baud rate.")
        print("  Power off? Wrong port? Bus not terminated at the first servo?")
        return set(), False

    ids: set[int] = set()
    baud_ok = True
    for baud, id_list in sorted(found.items()):
        mark = "" if baud == EXPECTED_BAUD else "   <- NOT the baud rate lerobot uses"
        print(f"  {baud:>9} baud: {sorted(id_list)}{mark}")
        if baud != EXPECTED_BAUD:
            baud_ok = False
        ids |= set(id_list)

    print()
    width = max(len(n) for n in expected.values())
    for mid, name in sorted(expected.items()):
        print(f"    id {mid:<2}  {name:<{width}}  {'present' if mid in ids else 'MISSING'}")

    extra = sorted(ids - set(expected))
    if extra:
        print(f"\n    unexpected ids: {extra}")
        print("    Another servo on this bus, or one left on a stale id.")

    return ids, baud_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port1", default=DEFAULT_PORT1, help="left arm + head bus")
    ap.add_argument("--port2", default=DEFAULT_PORT2, help="right arm + base bus")
    args = ap.parse_args()

    ids1, baud1 = scan(args.port1, BUS1, "BUS 1   left arm (1-6) + head (7-8)")
    ids2, baud2 = scan(args.port2, BUS2, "BUS 2   right arm (1-6) + base wheels (7-9)")

    arms = {1, 2, 3, 4, 5, 6}
    head = {7, 8}
    base = {7, 8, 9}

    have_arms = arms <= ids1 and arms <= ids2
    have_head = head <= ids1
    have_base = base <= ids2

    print(f"\n{'=' * 68}\nWHAT YOU CAN RUN\n{'=' * 68}")
    print(f"  both arms  {'yes' if have_arms else 'no'}")
    print(f"  head       {'yes' if have_head else 'no'}")
    print(f"  base       {'yes' if have_base else 'no'}")
    print()

    if have_arms and have_head and have_base:
        print("  Full xlerobot. Calibrate it:")
        print("    lerobot-calibrate --robot.type=xlerobot --robot.id=xlerobot \\")
        print(f"      --robot.port1={args.port1} \\")
        print(f"      --robot.port2={args.port2}")
        print("\n  Then: ./config/cart-selftest.sh")
    elif have_arms:
        missing = []
        if not have_head:
            missing.append(f"head (bus1 ids {sorted(head - ids1)})")
        if not have_base:
            missing.append(f"base (bus2 ids {sorted(base - ids2)})")
        print(f"  Arms only - {', '.join(missing)} not responding.")
        print("  --robot.type=xlerobot will fail on connect: it reads every")
        print("  motor unconditionally, including base velocity.")
        print("\n  Use the arms-only path until those are wired:")
        print("    config/bi-arms.yaml, config/cart-remote-arms.yaml")
        print("\n  To bring a new servo onto a bus, set its id first:")
        print("    lerobot-setup-motors --robot.type=xlerobot --robot.port1=... --robot.port2=...")
    else:
        print("  Arms incomplete. Fix that before anything else.")
        print("  A whole bus silent is usually the cable or power; one id")
        print("  silent is usually that servo or its daisy-chain link.")

    if not (baud1 and baud2):
        print("\n  WARNING: something answered at a non-default baud rate.")
        print(f"  lerobot talks at {EXPECTED_BAUD}. Re-run lerobot-setup-motors for those.")

    return 0 if (have_arms and have_head and have_base) else 1


if __name__ == "__main__":
    sys.exit(main())
