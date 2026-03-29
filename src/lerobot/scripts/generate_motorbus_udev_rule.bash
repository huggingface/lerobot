#!/bin/bash
# Copyright 2024 The LeRobot Authors
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
set -euo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

error() {
  echo -e "[${RED}error${NC}]: ${1}" >&2
  exit 1
}

info() {
  echo -e "[${GREEN}info${NC}]: ${1}" >&2
}

usage() {
    echo "Usage: $0 -d <device> -n <symlink_name> [-h]"
    echo ""
    echo "Options:"
    echo "  -d <device>       TTY device to configure (required, e.g., /dev/ttyUSB0)"
    echo "  -n <name>         Desired symlink name (required)"
    echo "  -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d /dev/ttyACM0 -n leader_arm"
    echo "  $0 -d /dev/ttyACM1 -n follower_arm"
    exit 0
}

# Required parameters
TTY_DEVICE=""
SYMLINK_NAME=""

# Parse arguments using getopts
while getopts "d:n:h" opt; do
    case $opt in
        d) TTY_DEVICE="$OPTARG" ;;
        n) SYMLINK_NAME="$OPTARG" ;;
        h) usage ;;
        \?) error "Invalid option: -$OPTARG. Use -h for help." ;;
        :) error "Option -$OPTARG requires an argument." ;;
    esac
done

# Check required parameters
if [ -z "$TTY_DEVICE" ]; then
    error "TTY device is required. Use -d <device>. Use -h for help."
fi

if [ -z "$SYMLINK_NAME" ]; then
    error "Symlink name is required. Use -n <name>. Use -h for help."
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run with sudo to create udev rule"
fi

# Check if the device exists
if [ ! -e "$TTY_DEVICE" ]; then
    error "Device '$TTY_DEVICE' not found"
fi

# Validate symlink name (allow alphanumeric, hyphens, underscores)
if ! [[ "$SYMLINK_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    error "Invalid symlink name '$SYMLINK_NAME'. Use only letters, numbers, hyphens, and underscores."
fi

# Get device information
# Anchor grep with ^ and use -m1 to avoid partial matches on similar property names
# (e.g., ID_VENDOR_ID should not match a hypothetical ID_VENDOR_ID_FROM_DATABASE)
VENDOR=$(udevadm info -n "$TTY_DEVICE" -q property | grep -m1 '^ID_VENDOR_ID=' | cut -d= -f2)
PRODUCT=$(udevadm info -n "$TTY_DEVICE" -q property | grep -m1 '^ID_MODEL_ID=' | cut -d= -f2)
SERIAL=$(udevadm info -n "$TTY_DEVICE" -q property | grep -m1 '^ID_SERIAL_SHORT=' | cut -d= -f2)

info "Detected device information:"
echo "  Device: $TTY_DEVICE"
echo "  Vendor ID: $VENDOR"
echo "  Product ID: $PRODUCT"
echo "  Serial: $SERIAL"
echo ""

# Validate that we have required identifiers
if [ -z "$VENDOR" ] || [ -z "$PRODUCT" ] || [ -z "$SERIAL" ]; then
  error "Could not detect all required identifiers for device $TTY_DEVICE:
  Vendor ID: ${VENDOR:-NOT FOUND}
  Product ID: ${PRODUCT:-NOT FOUND}
  Serial: ${SERIAL:-NOT FOUND}"
fi

# Create udev rule
UDEV_RULE="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VENDOR\", ATTRS{idProduct}==\"$PRODUCT\", ATTRS{serial}==\"$SERIAL\", MODE=\"0666\", SYMLINK+=\"$SYMLINK_NAME\""

# Define the udev rule file path
UDEV_RULE_FILE="/etc/udev/rules.d/99-tty-$SYMLINK_NAME.rules"

# Save the udev rule
# Yes, udev rules support '#' comments — lines starting with '#' are ignored by udev
{
    echo "# MotorBus udev rule for $SYMLINK_NAME"
    echo "# Generated on $(date)"
    echo "# Device: $TTY_DEVICE -> /dev/$SYMLINK_NAME"
    echo ""
    echo "$UDEV_RULE"
} > "$UDEV_RULE_FILE" || error "Failed to save udev rule"

info "Udev rule saved successfully to $UDEV_RULE_FILE"

# Set proper permissions
chmod 644 "$UDEV_RULE_FILE"

# Reload udev rules
info "Reloading udev rules..."
udevadm control --reload-rules && udevadm trigger

echo ""
info "Setup complete! Your device should now be available as '/dev/$SYMLINK_NAME'"
echo ""
info "To remove this rule later, delete: $UDEV_RULE_FILE"
