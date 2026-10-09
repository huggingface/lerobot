#!/usr/bin/env bash
# Run AT THE OPERATOR STATION (rosie, the Pi). Serves the leader arms and
# gamepad to the cart.
#
#   ./config/operator-leader-host.sh
#   ./config/operator-leader-host.sh --host.rate_hz=90
#
# Start this FIRST. The cart's connect() waits for the first action and will
# not invent one - that first action carries the leaders' current pose and
# the followers move to it.
#
# Nothing about the robot lives here. This process reads two serial buses
# and a HID device and publishes actions; the cart owns everything else.
#
# SEEING WHAT THE ROBOT SEES
# The video comes from the cart's own display stream, not from this process.
# Leave a viewer running here and start the cart with --watch:
#
#   rerun --port 9876                 # this machine, then on the cart:
#   ./config/cart-teleop.sh --watch   #   pushes to rosie:9876
#
# (Check `rerun --help` for your version's listen flag; the viewer has to be
# accepting gRPC connections for the cart to push to it.)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG="${CONFIG:-$REPO/config/operator-leader-host.yaml}"
LEFT_LEADER="${LEFT_LEADER:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90149222-if00}"
RIGHT_LEADER="${RIGHT_LEADER:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A017922-if00}"

require_dev "$LEFT_LEADER"  "left leader arm"
require_dev "$RIGHT_LEADER" "right leader arm"

if ! ls /dev/input/js* >/dev/null 2>&1; then
  echo "warning: no /dev/input/js* - the gamepad is not attached." >&2
  echo "         The arms will still work; the head and base will not." >&2
fi

echo "config : $CONFIG"
echo "serving: tcp://$(hostname -I 2>/dev/null | awk '{print $1}'):5557"
echo
grep -E '^(remap_arm_prefix|emit_head|emit_base|emit_y_vel):' "$CONFIG" | sed 's/^/         /'
echo "         ^ these must match the cart's config. A mismatch is silent."
echo

exec "${RUN[@]}" python -m lerobot_teleoperator_xlerobot_leader_remote.leader_host \
  --config_path="$CONFIG" "$@"
