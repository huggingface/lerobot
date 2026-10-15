#!/usr/bin/env bash
# Soak the leader link for a fixed time and grade it. See soak.py's docstring.
#
#   ./config/leader-soak.sh                     5 min, quiet
#   ./config/leader-soak.sh --minutes 10        longer
#   ./config/leader-soak.sh --ip 127.0.0.1      on the operator station itself
#
# Start ./config/operator-leader-host.sh on rosie first. Run it once quiet,
# then once more with ./config/video-test.sh <mac-ip> going in a second
# terminal on the cart - that is the traffic that used to stall the link.
# Both must print PASS.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

IP_DEFAULT="192.168.1.100"   # rosie
if [[ "${1:-}" != --ip* ]]; then
  set -- --ip "$IP_DEFAULT" "$@"
fi

exec "${RUN[@]}" python -m lerobot_teleoperator_xlerobot_leader_remote.soak "$@"
