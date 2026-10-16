#!/usr/bin/env python3
"""Give ONE new STS3215 its id and lerobot's baud rate. Nothing else is touched.

    ./config/servo.sh set-id --bus 2 --id 5      right wrist roll
    ./config/servo.sh set-id --bus 1 --id 7      head_motor_1, and so on

    # on rosie, on a leader arm's bus board (its leader unplugged):
    ./config/servo.sh set-id --bus 2 --id 5 --port /dev/serial/by-id/usb-1a86_...

Any bus board will do - the id lives in the servo, not the board. A
leader board supplies a lower voltage than the followers', which is fine
for this: a 12 V servo answers and takes its id at 5 V. (The reverse - a
5-7.4 V servo on a follower board - is not fine; it will burn.)

lerobot-setup-motors walks every servo on the robot, one at a time, asking
you to re-plug each in turn. Replacing one servo needs only this.

THE NEW SERVO MUST BE THE ONLY THING ON THE BUS. Unplug the arm's chain
from the bus board and connect the new servo to it directly. A servo fresh
from the factory is id 1 - and so is the shoulder pan already on that bus,
so with the arm still connected this would renumber whichever answered
first. The script refuses to run unless exactly one servo answers.

After it: plug the arm back in, fit the servo, then recalibrate that arm -
the new servo's homing offset is not the old one's.
"""

import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .hw import JOINTS as NAMES, PORTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", type=int, choices=(1, 2), required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--port", default=None,
                    help="any bus board to do it on - e.g. a leader arm's board on rosie. "
                         "Default: that bus's follower board on elroy. --bus still picks "
                         "which joint name and id table to use.")
    args = ap.parse_args()

    if args.id not in NAMES[args.bus]:
        print(f"error: bus {args.bus} has no id {args.id}; it has {sorted(NAMES[args.bus])}", file=sys.stderr)
        return 2
    name = NAMES[args.bus][args.id]
    port = args.port or PORTS[args.bus]

    print(f"\nSetting up {name}: bus {args.bus}, id {args.id}")
    print(f"  {port}\n")
    print("  Unplug the arm's servo chain from this bus board, and connect ONLY")
    print("  the new servo to it. Servo power on.")
    try:
        input("\n  ENTER when the new servo is the only one connected, Ctrl-C to stop: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return 1

    bus = FeetechMotorsBus(port=port, motors={name: Motor(args.id, "sts3215", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    try:
        # Look at every baud rate first, and insist on exactly one servo -
        # lerobot's own search takes the first one that answers.
        found: list[tuple[int, int, int]] = []
        for baud in bus.model_baudrate_table["sts3215"]:
            bus.set_baudrate(baud)
            for mid, model in (bus.broadcast_ping() or {}).items():
                found.append((baud, mid, model))
        if not found:
            print("\n  Nothing answered at any baud rate. Check the cable and that servo power is on.")
            return 1
        if len(found) > 1:
            print("\n  More than one servo answered - the arm is still connected:")
            for baud, mid, model in found:
                print(f"    id {mid} at {baud} baud (model {model})")
            print("  Disconnect everything but the new servo and run this again. Nothing was written.")
            return 1

        baud, mid, model = found[0]
        expected = bus.model_number_table["sts3215"]
        print(f"\n  Found one servo: id {mid} at {baud} baud, model number {model}.")
        if model != expected:
            print(f"  That is not an STS3215 (expected model {expected}). Nothing was written.")
            return 1
        if mid == args.id and baud == bus.default_baudrate:
            print(f"  It is already id {args.id} at {baud} baud. Nothing to do.")
            return 0

        bus.setup_motor(name, initial_baudrate=baud, initial_id=mid)

        bus.set_baudrate(bus.default_baudrate)
        check = bus.broadcast_ping() or {}
        if args.id in check:
            print(f"  Done: now id {args.id} at {bus.default_baudrate} baud.")
        else:
            print(f"  Wrote it, but a re-scan at {bus.default_baudrate} baud found {check}. Run this again.")
            return 1
    finally:
        bus.disconnect(disable_torque=False)

    print("\n  Next:")
    print("    1. Plug the arm's chain back into the bus board and fit the new servo.")
    print("    2. ./config/servo.sh scan       - every id should be present")
    print("    3. Recalibrate this arm (the new servo's offset is not the old one's).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
