#!/usr/bin/env bash
# Give one replacement servo its id and lerobot's baud rate. See set_servo_id.py.
#
#   ./config/set-servo-id.sh --bus 2 --id 5     right wrist roll
#
# The new servo must be the ONLY servo connected to that bus board.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUN[@]}" python "$REPO/config/set_servo_id.py" "$@"
