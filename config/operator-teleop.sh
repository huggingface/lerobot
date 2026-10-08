#!/usr/bin/env bash
# Run AT THE OPERATOR STATION (rosie, the Pi), after cart-host.sh is up on
# the cart.
#
#   ./config/operator-teleop.sh
#   ./config/operator-teleop.sh --display            rerun, on a local desktop
#   ./config/operator-teleop.sh --display=web        foxglove on :8765, any browser
#
# Leader arms drive the two arms; the gamepad drives the head and base. All
# 17 action dimensions are covered. See config/operator-client.yaml.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG="${CONFIG:-$REPO/config/operator-client.yaml}"
LEFT_LEADER="${LEFT_LEADER:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90149222-if00}"
RIGHT_LEADER="${RIGHT_LEADER:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A017922-if00}"

require_dev "$LEFT_LEADER"  "left leader arm"
require_dev "$RIGHT_LEADER" "right leader arm"

if ! ls /dev/input/js* >/dev/null 2>&1; then
  echo "warning: no /dev/input/js* - the gamepad is not attached." >&2
  echo "         The arms will still work; the head and base will not." >&2
fi

HOST_IP="$(awk -F'[ #]+' '/remote_ip:/ {print $3; exit}' "$CONFIG")"
echo "config : $CONFIG"
echo "cart   : $HOST_IP"
if ! ping -c1 -W2 "$HOST_IP" >/dev/null 2>&1; then
  echo "warning: $HOST_IP does not answer ping - is cart-host.sh running?" >&2
fi
echo

# Video. On a local desktop, rerun opens one window with a panel per camera -
# three feeds, not three OS windows; lerobot has no three-window mode. Over
# SSH use foxglove and view it in a browser instead, since rerun needs a
# display.
DISPLAY_ARGS=()
case "${1:-}" in
  --display)
    shift
    if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
      echo "error: --display needs a desktop session; none detected." >&2
      echo "       Over SSH use --display=web instead." >&2
      exit 1
    fi
    DISPLAY_ARGS=(--display_data=true --display_mode=rerun)
    ;;
  --display=web)
    shift
    DISPLAY_ARGS=(--display_data=true --display_mode=foxglove
                  --display_ip=0.0.0.0 --display_port=8765)
    echo "foxglove: ws://$(hostname).local:8765"
    ;;
esac

exec run lerobot-teleoperate --config_path="$CONFIG" "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}" "$@"
