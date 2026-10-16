"""./config/servo.sh <command> [args]  -  the cart's servo tools in one place.

    scan     which ids answer on each bus, at which baud rate
    power    voltage, temperature and load per servo, torque on one at a time
    regs     every register side by side on one bus; flags the odd one out
    set-id   give ONE replacement servo its id and baud rate
    recal    recalibrate one joint after a servo swap, leaving the rest

./config/servo.sh <command> --help for each.
"""

import importlib
import sys

COMMANDS = {
    "scan": "scan",
    "power": "power",
    "regs": "regs",
    "set-id": "set_id",
    "recal": "recal",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 2
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"unknown command '{cmd}'. One of: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"servo_tools.{COMMANDS[cmd]}")
    sys.argv = [f"servo.sh {cmd}", *sys.argv[2:]]   # argparse's prog
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
