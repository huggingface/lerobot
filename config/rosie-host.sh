#!/usr/bin/env bash
# Run ON ROSIE (the Pi). Serves the robot over ZMQ for a remote client.
#
#   ./scripts/rosie-host.sh
#
# Any extra arguments are passed through to the host, so you can override
# anything:  ./scripts/rosie-host.sh --host.max_loop_freq_hz=20
#
# Requires the full xlerobot motor layout, not just the arms:
#   bus1  left arm 1-6  + head 7-8
#   bus2  right arm 1-6 + base wheels 7-9
# Missing motors fail with "no status packet" naming the id.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The robot config lives in a YAML because the host's individual flags cannot
# express cameras. Set HOST_CONFIG= to override, or NO_CAMERAS=1 to fall back
# to the bare flags (joint state only - useful to isolate a camera problem).
HOST_CONFIG="${HOST_CONFIG:-$REPO/config/rosie-host.yaml}"
ROBOT_ID="${ROBOT_ID:-rosie}"
PORT1="${PORT1:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00}"  # left bus
PORT2="${PORT2:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00}"  # right bus
ZMQ_CMD="${ZMQ_CMD:-5555}"
ZMQ_OBS="${ZMQ_OBS:-5556}"

require_dev "$PORT1" "left arm bus (port1)"
require_dev "$PORT2" "right arm bus (port2)"

echo "host   : $ROBOT_ID"
echo "port1  : $PORT1"
echo "port2  : $PORT2"
echo "zmq    : cmd $ZMQ_CMD, observations $ZMQ_OBS"
echo "listen : $(hostname -I 2>/dev/null | awk '{print $1}')"
echo

if [[ -n "${NO_CAMERAS:-}" ]]; then
  echo "cameras: disabled (NO_CAMERAS set)"
  echo
  exec run python -m lerobot_robot_xlerobot.xlerobot_host \
    --robot.id="$ROBOT_ID" \
    --robot.port1="$PORT1" \
    --robot.port2="$PORT2" \
    --host.port_zmq_cmd="$ZMQ_CMD" \
    --host.port_zmq_observations="$ZMQ_OBS" \
    "$@"
fi

echo "cameras: from $HOST_CONFIG"
echo
exec run python -m lerobot_robot_xlerobot.xlerobot_host \
  --config_path="$HOST_CONFIG" \
  --host.port_zmq_cmd="$ZMQ_CMD" \
  --host.port_zmq_observations="$ZMQ_OBS" \
  "$@"
