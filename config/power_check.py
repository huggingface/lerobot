#!/usr/bin/env python
"""Ask the servos what voltage they are actually seeing.

"There is no status packet" from one motor looks the same whether the cause
is a loose connector or a supply that sags when the last few servos
energize. The servos themselves can tell you which: every STS3215 reports
its own bus voltage, temperature and load.

    uv run --project ../XLeRobot python config/power_check.py

Three phases:

  1. everything relaxed      - baseline voltage per motor
  2. torque enabled ONE AT A TIME, re-reading voltage after each
  3. everything holding      - the worst case

If the voltage falls as motors are added, and falls furthest at the end of
a chain, the supply or the wiring is the problem. If it stays flat and one
motor simply stops answering, it is that motor or its connector.

Torque is left DISABLED at the end, so the arms will be limp - support them.
"""

from __future__ import annotations

import argparse
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

M = MotorNormMode.RANGE_M100_100

BUS1 = {
    "left_arm_shoulder_pan": 1, "left_arm_shoulder_lift": 2, "left_arm_elbow_flex": 3,
    "left_arm_wrist_flex": 4, "left_arm_wrist_roll": 5, "left_arm_gripper": 6,
    "head_motor_1": 7, "head_motor_2": 8,
}
BUS2 = {
    "right_arm_shoulder_pan": 1, "right_arm_shoulder_lift": 2, "right_arm_elbow_flex": 3,
    "right_arm_wrist_flex": 4, "right_arm_wrist_roll": 5, "right_arm_gripper": 6,
    "base_left_wheel": 7, "base_back_wheel": 8, "base_right_wheel": 9,
}

P1 = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00"
P2 = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00"

# STS3215 reports voltage in tenths of a volt. Nominal supply is 12 V; the
# datasheet floor is around 9. Current is in raw counts - the scaling is not
# documented consistently, so it is shown unconverted and used only to
# compare motors with each other.
def volts(raw): return raw / 10.0


def read1(bus, name, reg):
    try:
        return bus.read(reg, name, num_retry=2)
    except Exception:
        return None


def table(bus, motors, label):
    print(f"\n  {label}")
    print(f"    {'motor':<26}{'volts':>8}{'temp C':>8}{'load':>8}{'current':>9}")
    lo, lo_name = None, None
    for name in motors:
        v = read1(bus, name, "Present_Voltage")
        t = read1(bus, name, "Present_Temperature")
        ld = read1(bus, name, "Present_Load")
        cu = read1(bus, name, "Present_Current")
        if v is None:
            print(f"    {name:<26}{'NO ANSWER':>8}")
            continue
        if lo is None or v < lo:
            lo, lo_name = v, name
        print(f"    {name:<26}{volts(v):>8.1f}{t if t is not None else '-':>8}"
              f"{ld if ld is not None else '-':>8}{cu if cu is not None else '-':>9}")
    if lo is not None:
        print(f"    lowest: {volts(lo):.1f} V at {lo_name}")
    return lo


def progressive(bus, motors, label):
    """Enable torque one motor at a time, watching the voltage fall."""
    print(f"\n  {label} - enabling torque one at a time")
    print(f"    {'after enabling':<26}{'its volts':>11}{'bus min':>10}")
    worst = None
    for name in motors:
        try:
            bus.enable_torque([name], num_retry=2)
        except Exception as e:
            print(f"    {name:<26}{'FAILED':>11}   {type(e).__name__}")
            print(f"      {e}")
            continue
        time.sleep(0.25)          # let the holding current settle
        v = read1(bus, name, "Present_Voltage")
        allv = [x for x in (read1(bus, n, "Present_Voltage") for n in motors) if x]
        mn = min(allv) if allv else None
        if mn is not None and (worst is None or mn < worst):
            worst = mn
        print(f"    {name:<26}{volts(v) if v else '-':>11}"
              f"{volts(mn) if mn else '-':>10.1f}")
    return worst


def run(port, motors, label):
    bus = FeetechMotorsBus(port, {n: Motor(i, "sts3215", M) for n, i in motors.items()})
    print(f"\n{'=' * 64}\n{label}\n  {port}\n{'=' * 64}")
    try:
        bus.connect(handshake=False)   # a flaky motor must not stop the survey
    except Exception as e:
        print(f"  could not open: {e}")
        return None

    try:
        bus.disable_torque(num_retry=2)
    except Exception as e:
        print(f"  (could not relax everything first: {e})")
    time.sleep(0.3)

    rest = table(bus, motors, "1. relaxed")
    worst = progressive(bus, motors, "2. progressive")
    held = table(bus, motors, "3. all holding")

    try:
        bus.disable_torque(num_retry=2)
    except Exception:
        pass
    bus.disconnect()
    return rest, worst, held


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port1", default=P1)
    ap.add_argument("--port2", default=P2)
    args = ap.parse_args()

    print(__doc__.strip().splitlines()[0])
    print("\nArms will go limp at the end. Support them before continuing.")
    try:
        input("ENTER to start, Ctrl-C to abandon: ")
    except (EOFError, KeyboardInterrupt):
        return 1

    r1 = run(args.port1, BUS1, "BUS 1   left arm 1-6 + head 7-8")
    r2 = run(args.port2, BUS2, "BUS 2   right arm 1-6 + base wheels 7-9")

    print(f"\n{'=' * 64}\nREADING THIS\n{'=' * 64}")
    for label, r in (("bus1", r1), ("bus2", r2)):
        if not r:
            continue
        rest, worst, held = r
        if rest and worst:
            print(f"  {label}: {volts(rest):.1f} V relaxed -> {volts(worst):.1f} V worst "
                  f"(sag {volts(rest) - volts(worst):.1f} V)")
    print("""
  A sag of a few tenths is normal. A volt or more, or any motor below
  about 9 V, is a supply that cannot hold up the whole robot - and the
  motor that stops answering will be the one at the end of the longest
  cable run, which is why it looks like a bad servo.

  Flat voltage with one motor silent means that motor or its connector,
  not the supply.

  Rising temperature on one motor while its neighbours stay cool means it
  is fighting something mechanical.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
