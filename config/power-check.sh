#!/usr/bin/env bash
# Ask the servos what voltage they are seeing, with torque applied
# progressively. Run it when a motor stops answering, to tell a sagging
# supply apart from a bad connector.
#
#   ./config/power-check.sh
#
# Arms are left limp at the end. Support them.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUN[@]}" python "$REPO/config/power_check.py" "$@"
