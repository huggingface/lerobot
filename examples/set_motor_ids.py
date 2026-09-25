# !/usr/bin/env python

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

"""Set motor IDs for individual SO-101 leader arm motors.

This script is useful when `lerobot-setup-motors` fails partway through
(e.g. due to a loose cable), leaving some motors unconfigured. Since motor
IDs are stored in non-volatile memory, already-configured motors are fine —
only the remaining ones need to be set.

Example usage:

    # Configure all motors (same as lerobot-setup-motors, but resumable)
    python set_motor_ids.py --port /dev/tty.usbmodem5B790332241

    # Configure only specific motors after a partial failure
    python set_motor_ids.py --port /dev/tty.usbmodem5B790332241 --motors shoulder_lift shoulder_pan
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

LEADER_MOTORS: dict[str, Motor] = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set motor IDs for individual SO-101 leader arm motors. "
            "Useful for resuming a failed lerobot-setup-motors run."
        )
    )
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port of the URT-2 / bus servo adapter (e.g. /dev/tty.usbmodem5B790332241)",
    )
    parser.add_argument(
        "--motors",
        type=str,
        nargs="+",
        choices=list(LEADER_MOTORS.keys()),
        default=list(LEADER_MOTORS.keys()),
        help=(
            "One or more motor names to configure. Defaults to all motors in reverse order "
            "(gripper first, shoulder_pan last), matching the lerobot-setup-motors behaviour. "
            "Choices: " + ", ".join(LEADER_MOTORS.keys())
        ),
    )
    args = parser.parse_args()

    selected: dict[str, Motor] = {name: LEADER_MOTORS[name] for name in args.motors}
    bus = FeetechMotorsBus(port=args.port, motors=selected)

    for motor in reversed(list(selected.keys())):
        input(f"Connect the controller board to the '{motor}' motor only and press Enter...")
        bus.setup_motor(motor)
        print(f"'{motor}' motor id set to {selected[motor].id}")

    print("Done! All requested motors configured.")


if __name__ == "__main__":
    main()
