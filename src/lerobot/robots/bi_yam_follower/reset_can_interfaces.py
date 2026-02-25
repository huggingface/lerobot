#!/usr/bin/env python3

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
Script to reset all CAN interfaces with the correct bitrate for Yam arms.

This script automatically detects all CAN interfaces on your system and resets them
with the required bitrate of 1000000 for Yam arm communication.

Example usage:
    python src/lerobot/robots/bi_yam_follower/reset_can_interfaces.py
"""

import os
import re
import subprocess
import sys


def is_root():
    """Check if the script is running with root privileges."""
    return os.geteuid() == 0


def get_sudo_prefix():
    """Return 'sudo' if not running as root, otherwise empty string."""
    return "" if is_root() else "sudo"


def get_can_interfaces():
    """Get all CAN interfaces from the system."""
    try:
        result = subprocess.run(  # nosec B607
            ["ip", "link", "show"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract CAN interface names (e.g., can0, can1, can_follower_r, etc.)
        can_interfaces = re.findall(r"(?<=: )(can\w+)", result.stdout)
        return can_interfaces
    except subprocess.CalledProcessError as e:
        print(f"error getting can interfaces: {e}")
        return []


def reset_can_interface(interface, bitrate=1000000):
    """Reset a CAN interface with the specified bitrate."""
    sudo = get_sudo_prefix()

    try:
        # Bring interface down
        subprocess.run(  # nosec B607
            f"{sudo} ip link set {interface} down".split(),
            check=True,
        )

        # Bring interface up with bitrate
        subprocess.run(  # nosec B607
            f"{sudo} ip link set {interface} up type can bitrate {bitrate}".split(),
            check=True,
        )

        print(f"  [{interface}] configured with bitrate {bitrate}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  [{interface}] error: {e}")
        return False


def main():
    """Main function to reset all CAN interfaces."""
    print("resetting can interfaces for yam arms...\n")

    # Get all CAN interfaces
    can_interfaces = get_can_interfaces()

    if not can_interfaces:
        print("error: no can interfaces found on this system")
        print("\nplease check:")
        print("  - can hardware is properly connected")
        print("  - can drivers are loaded (e.g., 'modprobe can')")
        print("  - persistent can interface names are configured")
        sys.exit(1)

    print(f"detected can interfaces: {', '.join(can_interfaces)}")
    print("configuring with bitrate 1000000...\n")

    # Reset each interface
    success_count = 0
    for interface in can_interfaces:
        if reset_can_interface(interface):
            success_count += 1

    print()
    if success_count == len(can_interfaces):
        print(f"successfully reset all {success_count} can interfaces")
    else:
        print(f"warning: only reset {success_count}/{len(can_interfaces)} interfaces")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\ninterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nerror: {e}")
        sys.exit(1)
