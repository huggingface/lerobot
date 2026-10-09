#!/usr/bin/env bash
# Watch the leader stream with no robot attached.
#
#   ./config/leader-tap.sh                 from the cart, over the network
#   ./config/leader-tap.sh --ip 127.0.0.1  on the operator station itself
#
# Start ./config/operator-leader-host.sh first.
#
# This exercises the ZMQ link, the client teleoperator and the action keys,
# and shows whether moving a leader actually changes what arrives - with
# the robot left out, which is the part that is expensive to get wrong.
#
# Run it on the operator station first to take the network out of the
# question, then from the cart to put it back in. If it works locally and
# not remotely, the problem is the link, not the leaders.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

IP_DEFAULT="192.168.1.100"   # rosie
if [[ "${1:-}" != --ip* ]]; then
  set -- --ip "$IP_DEFAULT" "$@"
fi

exec "${RUN[@]}" python -m lerobot_teleoperator_xlerobot_leader_remote.tap "$@"
