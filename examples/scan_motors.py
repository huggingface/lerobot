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

"""Scan the motor bus and report which motor IDs are present.

Useful when motor labels have been lost or you need to verify which
motors have already been configured.

Connect one motor at a time and run this script to identify its ID,
or connect all motors on the daisy chain to get a full report.

Example usage:

    # Scan for all motors (IDs 1-6)
    python scan_motors.py --port /dev/tty.usbmodem5B790332241

    # Scan a custom ID range
    python scan_motors.py --port /dev/tty.usbmodem5B790332241 --min-id 1 --max-id 253
"""

import argparse

import scservo_sdk as scs

LEADER_MOTOR_NAMES: dict[int, str] = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan the motor bus and report which motor IDs are present."
    )
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port of the URT-2 / bus servo adapter (e.g. /dev/tty.usbmodem5B790332241)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=1000000,
        help="Baud rate to use for scanning (default: 1000000)",
    )
    parser.add_argument(
        "--min-id",
        type=int,
        default=1,
        help="Lowest motor ID to scan (default: 1)",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=6,
        help="Highest motor ID to scan (default: 6)",
    )
    args = parser.parse_args()

    port_handler = scs.PortHandler(args.port)
    packet_handler = scs.PacketHandler(0)  # protocol version 0 for Feetech STS3215

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open port {args.port}")
    if not port_handler.setBaudRate(args.baudrate):
        raise RuntimeError(f"Failed to set baudrate {args.baudrate}")

    print(f"Scanning IDs {args.min_id}–{args.max_id} on {args.port} at {args.baudrate} baud...\n")

    found = []
    for motor_id in range(args.min_id, args.max_id + 1):
        _, result, _ = packet_handler.ping(port_handler, motor_id)
        if result == scs.COMM_SUCCESS:
            joint = LEADER_MOTOR_NAMES.get(motor_id, "unknown joint")
            print(f"  ✓ Found motor ID {motor_id}  →  {joint}")
            found.append(motor_id)

    port_handler.closePort()

    print(f"\nFound {len(found)} motor(s): {found}" if found else "\nNo motors found. Check cable connection.")

    missing = [i for i in range(1, 7) if i not in found]
    if missing:
        print(f"Missing SO-101 IDs:  {missing}")
        print("Run set_motor_ids.py to configure the missing motors.")


if __name__ == "__main__":
    main()
