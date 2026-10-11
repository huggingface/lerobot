#!/usr/bin/env bash
# The ODESC on the bench: report, apply the bus limits, or watch it through
# a power cut. See config/odrive_check.py. Needs no motors.
#
#   ./config/odrive-check.sh            report config vs the cart's limits
#   ./config/odrive-check.sh --apply    write them, save, reboot, verify
#   ./config/odrive-check.sh --watch    the XH-M609 relay-drop test

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! lsusb 2>/dev/null | grep -q "1209:0d3"; then
  echo "error: no ODrive on USB (lsusb shows no 1209:0d3x)." >&2
  echo "       The board's USB only works with DC power on its terminals." >&2
  echo "       0d31 would be DFU mode - also wrong." >&2
  exit 1
fi

exec "${RUN[@]}" python "$REPO/config/odrive_check.py" "$@"
