#!/usr/bin/env bash
# Run ON THE CART (elroy, the Orin). Drives the robot from the leaders at
# the operator station, recording nothing.
#
#   ./config/cart-teleop.sh                 the full robot, headless
#   ./config/cart-teleop.sh --watch         + video to a rerun viewer
#   ./config/cart-teleop.sh --watch=web     + video as Foxglove, any browser
#   ./config/cart-teleop.sh --arms          arms only (see the refusal in _common.sh)
#
# Flags in any order. The full robot is the default because the arms-only
# config can destroy this machine's calibration - see guard_arms_config.
#
# Start ./config/operator-leader-host.sh on the Pi FIRST.
#
# Everything the robot has is local to this machine. Only the leaders are
# remote, and only their actions cross the wifi.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"

warn_if_cpu_torch

# --full and --watch in any order. The old code tested $1 for --full, then
# $1 for --watch, so `--watch --full` silently fell through to the ARMS
# config - which on elroy means bi_so_follower, no calibration, and a fresh
# one started over the top of the robot's. An argument order that destroys
# a calibration is not an argument order.
#
# The full robot is also now the DEFAULT. --arms is the opt-in, because the
# dangerous choice should be the one you have to ask for.
CONFIG="$REPO/config/cart-remote.yaml"
ARGS=()
for a in "$@"; do
  case "$a" in
    --full)  CONFIG="$REPO/config/cart-remote.yaml" ;;
    --arms)  CONFIG="$REPO/config/cart-remote-arms.yaml" ;;
    *)       ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
guard_arms_config "$CONFIG"
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00" "left follower (bus1)"
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00" "right follower (bus2)"

OPERATOR="$(awk -F'[ #]+' '/remote_ip:/ {print $3; exit}' "$CONFIG")"
echo "config  : $CONFIG"
echo "leaders : $OPERATOR:5557"
if ! ping -c1 -W2 "$OPERATOR" >/dev/null 2>&1; then
  echo "warning: $OPERATOR does not answer ping - is operator-leader-host.sh running?" >&2
fi

parse_watch_all "$@"; set -- "${REMAINING[@]+"${REMAINING[@]}"}"
echo

exec "${RUN[@]}" lerobot-teleoperate --config_path="$CONFIG" \
  "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}" "$@"
