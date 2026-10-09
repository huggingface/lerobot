#!/usr/bin/env bash
# Run ON THE CART (elroy, the Orin). Records a dataset while the operator
# drives from the Pi.
#
#   ./config/cart-record.sh <repo_id> "<task description>"
#   ./config/cart-record.sh --watch carlkesselman/xlerobot-pick "Pick the block, drop it in the box"
#   ./config/cart-record.sh --full --watch me/ds "task"
#
# Flags, in this order, before the repo id:
#   --full      arms + head + base (needs those motors; default is arms only)
#   --watch     video to the operator station - see config/_video.sh for
#               --watch=web and --watch=local
#
# Everything else passes through:
#   ./config/cart-record.sh me/ds "task" --dataset.num_episodes=10
#
# WHAT CROSSES THE WIFI, AND WHAT DOES NOT
#
# Only the leader actions, one direction, plus the operator's video if you
# ask for it. The followers and all three cameras are wired to THIS machine,
# so every frame goes from USB straight into the dataset at full rate. There
# is no conflated socket between the sensors and the disk, which is what
# would otherwise drop frames silently on each wifi hiccup.
#
# This machine is also the one that will run the policy, so the dataset is
# captured through exactly the camera pipeline, resolution and timing that
# inference will see. Record on one machine and infer on another and you get
# a train/serve skew you cannot observe.
#
# What the link costs instead is feel: wifi jitter lands in the action
# stream, and what the operator actually commanded is what gets recorded.
# The dataset stays self-consistent either way - but jerky input makes jerky
# demonstrations. Watch for "Leader link stalled" in the log; past
# stale_after_ms the base is zeroed and the arms hold.
#
# EPISODE CONTROL
# lerobot's keys (right arrow = end episode, left = re-record, escape =
# stop) are read by the process running here. Drive this script from an SSH
# session opened FROM the Pi and they work from the operator's keyboard -
# lerobot falls back to a terminal listener when pynput cannot capture, so
# an SSH session with a TTY is enough.
#
# UPLOAD
# push_to_hub defaults to true, so the dataset uploads when recording ends.
# Authenticate first, once:   hf auth login
# To stay local:              --dataset.push_to_hub=false

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"

warn_if_cpu_torch

CONFIG="$REPO/config/cart-remote-arms.yaml"
if [[ "${1:-}" == "--full" ]]; then
  shift
  CONFIG="$REPO/config/cart-remote.yaml"
fi
parse_watch "${1:-}" && shift
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

if [[ $# -lt 2 ]]; then
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 1
fi

REPO_ID="$1"; shift
TASK="$1"; shift

FPS="${FPS:-30}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-15}"

# The followers are here. The leaders are not, and must not be.
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00" "left follower (bus1)"
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00" "right follower (bus2)"

OPERATOR="$(awk -F'[ #]+' '/remote_ip:/ {print $3; exit}' "$CONFIG")"
if ! ping -c1 -W2 "$OPERATOR" >/dev/null 2>&1; then
  echo "warning: $OPERATOR does not answer ping - is operator-leader-host.sh running?" >&2
fi

echo "config  : $CONFIG"
echo "leaders : $OPERATOR:5557"
echo "dataset : $REPO_ID"
echo "task    : $TASK"
echo "episodes: $NUM_EPISODES x ${EPISODE_TIME_S}s (reset ${RESET_TIME_S}s) at ${FPS}fps"
echo

exec "${RUN[@]}" lerobot-record \
  --config_path="$CONFIG" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.fps="$FPS" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}" \
  "$@"
