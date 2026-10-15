#!/usr/bin/env bash
# Probe both Feetech buses and report which xlerobot motors answer.
#
#   ./config/scan-motors.sh
#   ./config/scan-motors.sh --port1 /dev/ttyACM0 --port2 /dev/ttyACM1
#
# Run it wherever the arms are plugged in right now - this is a hardware
# question, not a machine question. Nothing is written; it only pings.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

exec "${RUN[@]}" python "$REPO/config/scan_motors.py" "$@"
