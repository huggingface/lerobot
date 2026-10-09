#!/usr/bin/env bash
# How stale can a camera's frames get? See config/cam_probe.py.
#
#   ./config/cam-probe.sh /dev/cam_left
#   ./config/cam-probe.sh /dev/cam_left --buffersize 1

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[[ $# -ge 1 ]] || { echo "usage: $0 /dev/cam_left [--buffersize N]" >&2; exit 2; }
require_dev "$1" "camera"
exec "${RUN[@]}" python "$REPO/config/cam_probe.py" "$@"
