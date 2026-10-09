#!/usr/bin/env bash
# Run ON THE CART. Proves this machine's own hardware before anything else
# is layered on top: no network, no leaders, no dataset, no policy.
#
#   ./config/cart-selftest.sh                    read-only
#   ./config/cart-selftest.sh --head             + sweep the head
#   ./config/cart-selftest.sh --base             + drive the wheels (ON BLOCKS)
#   ./config/cart-selftest.sh --arms --head --base
#   ./config/cart-selftest.sh --no-cameras       isolate the servos
#
# Camera frames land in ./selftest-frames/. Look at them - a camera can be
# live, correctly sized and pointed at the wrong thing.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

warn_if_cpu_torch

CONFIG="${CONFIG:-$REPO/config/cart.yaml}"
PORT1="${PORT1:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00}"
PORT2="${PORT2:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00}"

require_dev "$PORT1" "left arm + head bus (port1)"
require_dev "$PORT2" "right arm + base bus (port2)"

for dev in /dev/cam_left /dev/cam_right; do
  if [[ ! -e "$dev" ]]; then
    echo "warning: $dev missing - the udev rule has not been set up on this" >&2
    echo "         machine, or a camera moved ports. Read the real path with:" >&2
    echo "           udevadm info -q property /dev/video0 | grep ID_PATH" >&2
  fi
done

exec "${RUN[@]}" python "$REPO/config/cart_selftest.py" --config_path="$CONFIG" "$@"
