#!/usr/bin/env bash
# The leader link with no robot attached. Start ./config/operator-leader-host.sh
# on rosie first.
#
#   ./config/leader-link.sh tap                 which action keys arrive, and
#                                               whether moving a leader changes them
#   ./config/leader-link.sh soak [--minutes 5]  timed for N minutes and graded:
#                                               stalls, lost messages, histogram.
#                                               Run quiet, then with
#                                               ./config/video.sh synthetic going.
#   ... --ip 127.0.0.1                          on rosie itself, network removed
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

IP_DEFAULT="192.168.1.100"   # rosie
cmd="${1:-}"
case "$cmd" in
  tap|soak) shift ;;
  *) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
if [[ "${1:-}" != --ip* ]]; then
  set -- --ip "$IP_DEFAULT" "$@"
fi
exec "${RUN[@]}" python -m "lerobot_teleoperator_xlerobot_leader_remote.$cmd" "$@"
