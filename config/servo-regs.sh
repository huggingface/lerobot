#!/usr/bin/env bash
# Every register on one bus, servo by servo; flags the odd one out.
# Read-only. See config/servo_regs.py.
#
#   ./config/servo-regs.sh            bus 2, right arm
#   ./config/servo-regs.sh --bus 1    bus 1, left arm + head
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUN[@]}" python "$REPO/config/servo_regs.py" "$@"
