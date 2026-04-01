# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
Scan a serial port for Feetech motors at different baud rates.

Use this to diagnose "Missing motor IDs" / "Full found motor list: {}" errors.
If motors respond at a different baud rate than 1Mbps, run lerobot-setup-motors
to configure them to the expected baud rate.

Example:

```shell
lerobot-scan-motors --port=/dev/tty.usbmodem5A7A0186071
```
"""

from dataclasses import dataclass

import draccus

from lerobot.motors.feetech import FeetechMotorsBus


@dataclass
class ScanConfig:
    port: str


@draccus.wrap()
def scan_motors(cfg: ScanConfig) -> None:
    print(f"Scanning port '{cfg.port}' for Feetech motors at all baud rates...")
    print("(This may take ~30 seconds)\n")

    try:
        baudrate_ids = FeetechMotorsBus.scan_port(cfg.port)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure the arm is powered on")
        print("  - Check USB cable and 3-pin motor cables")
        print("  - If using Waveshare controller, ensure jumpers are on 'B' channel (USB)")
        print("  - Verify the port with: lerobot-find-port")
        raise

    if not baudrate_ids:
        print("No motors found at any baud rate.")
        print("\nTroubleshooting:")
        print("  - Ensure the arm is powered on")
        print("  - Check USB cable and 3-pin motor cables (daisy chain)")
        print("  - If using Waveshare controller, ensure jumpers are on 'B' channel (USB)")
        print("  - Run lerobot-setup-motors if motors were never configured")
        return

    print("\nMotors found:")
    for baudrate, ids in sorted(baudrate_ids.items()):
        print(f"  {baudrate} bps: motor IDs {ids}")

    default_bps = 1_000_000
    if default_bps not in baudrate_ids:
        print(f"\n⚠ Motors did NOT respond at {default_bps} bps (expected default).")
        print("  Run lerobot-setup-motors to configure motors to 1Mbps:")
        print(f"    lerobot-setup-motors --robot.type=so101_follower --robot.port={cfg.port}")
    else:
        print(f"\n✓ Motors respond at {default_bps} bps. If you still get connection errors,")
        print("  check motor IDs (should be 1-6 for SO follower) and run lerobot-setup-motors if needed.")


def main() -> None:
    scan_motors()


if __name__ == "__main__":
    main()
