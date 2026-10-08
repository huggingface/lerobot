#!/usr/bin/env bash
# Run ON ELROY (the Orin), after rosie-host.sh is up on the Pi.
#
#   ./scripts/elroy-teleop.sh
#   ./scripts/elroy-teleop.sh --display_data=true     # any override passes through
#
# Drives the arms only: bi_so_leader supplies 12 of xlerobot's 17 action
# dimensions, so the head holds position and the base stays stopped. See
# config/elroy-client.yaml.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG="${CONFIG:-$REPO/config/elroy-client.yaml}"
LEFT_LEADER="${LEFT_LEADER:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90149222-if00}"
RIGHT_LEADER="${RIGHT_LEADER:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A017922-if00}"

require_dev "$LEFT_LEADER"  "left leader arm"
require_dev "$RIGHT_LEADER" "right leader arm"

HOST_IP="$(awk -F'[ #]+' '/remote_ip:/ {print $3; exit}' "$CONFIG")"
echo "config : $CONFIG"
echo "host    : $HOST_IP"
if ! ping -c1 -W2 "$HOST_IP" >/dev/null 2>&1; then
  echo "warning: $HOST_IP does not answer ping - is rosie-host.sh running?" >&2
fi
echo

exec run lerobot-teleoperate --config_path="$CONFIG" "$@"
