#!/usr/bin/env bash
# Run ON THE CART (elroy, the Orin). Drives the robot from the leaders at
# the operator station, recording nothing.
#
#   ./config/cart-teleop.sh                 the full robot, headless
#   ./config/cart-teleop.sh --watch         + video to a rerun viewer
#   ./config/cart-teleop.sh --watch=web     + video as Foxglove, any browser

#
# Start ./config/operator-leader-host.sh on the Pi FIRST.
#
# Everything the robot has is local to this machine. Only the leaders are
# remote, and only their actions cross the wifi.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"

warn_if_cpu_torch

CONFIG="$REPO/config/cart-remote.yaml"
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00" "left follower (bus1)"
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00" "right follower (bus2)"

OPERATOR="$(awk -F'[ #]+' '/remote_ip:/ {print $3; exit}' "$CONFIG")"
echo "config  : $CONFIG"
echo "leaders : $OPERATOR:5557"
if ! ping -c1 -W2 "$OPERATOR" >/dev/null 2>&1; then
  echo "warning: $OPERATOR does not answer ping - is operator-leader-host.sh running?" >&2
fi

# `|| :` is load-bearing. parse_watch_all returns 1 when there is no
# --watch flag, and under `set -e` a bare non-zero statement exits the
# script - silently, right after printing the header, which looks
# exactly like a program that started and stopped.
parse_watch_all "$@" || :
set -- "${REMAINING[@]+"${REMAINING[@]}"}"
echo

exec "${RUN[@]}" lerobot-teleoperate --config_path="$CONFIG" \
  "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}" "$@"
