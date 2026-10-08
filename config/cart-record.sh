#!/usr/bin/env bash
# Run ON THE CART (elroy, the Orin). Records a dataset from the two arms and
# their leaders, with every device local.
#
#   ./config/cart-record.sh <repo_id> "<task description>"
#   ./config/cart-record.sh carlkesselman/xlerobot-pick "Pick the block and drop it in the box"
#
#   --watch     also serve the live view to the operator station
#               (foxglove, ws://<cart>.local:8765)
#
# Everything else passes through:
#   ./config/cart-record.sh me/ds "task" --dataset.num_episodes=10 --dataset.episode_time_s=30
#
# WHY ON THE CART, AND NOT OVER THE LINK
#
# Two reasons, and the second is the one that matters now.
#
# 1. The ZMQ host sets CONFLATE on the observation socket: it keeps only the
#    newest frame and silently discards the rest. That is right for
#    teleoperation - you see current state, a slow link costs latency not
#    backlog - but recording across it drops frames whenever wifi hiccups,
#    with nothing in the logs to say so.
#
# 2. This machine is the one that will later run the policy. Recording here
#    means the dataset is captured through exactly the camera pipeline,
#    resolution and timing the policy will see at inference. Record on one
#    machine and infer on another and you have a train/serve skew you cannot
#    see.
#
# The price is that the leader arms have to be plugged into the cart while
# recording - this is a tethered session, not a driven-around one. Unplug
# them afterwards and the cart goes back to being autonomous hardware.
#
# Cameras run at 1280x720 here (config/bi-arms.yaml), not the 640x480 of
# cart-host.yaml: these frames never cross the air.
#
# UPLOAD
# push_to_hub defaults to true, so the dataset uploads when recording ends.
# Authenticate first, once:   hf auth login
# To stay local:              --dataset.push_to_hub=false

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ $# -lt 2 ]]; then
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 1
fi

REPO_ID="$1"; shift
TASK="$1"; shift

CONFIG="${CONFIG:-$REPO/config/bi-arms.yaml}"
FPS="${FPS:-30}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-15}"

WATCH_ARGS=()
if [[ "${1:-}" == "--watch" ]]; then
  shift
  WATCH_ARGS=(--display_data=true --display_mode=foxglove
              --display_ip=0.0.0.0 --display_port=8765
              --display_compressed_images=true)
  echo "watch   : open Foxglove on the operator station at  ws://$(hostname).local:8765"
fi

# Both followers and both leaders must be on THIS machine to record.
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00" "left follower"
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00" "right follower"
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90149222-if00" "left leader"
require_dev "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A017922-if00" "right leader"

echo "config  : $CONFIG"
echo "dataset : $REPO_ID"
echo "task    : $TASK"
echo "episodes: $NUM_EPISODES x ${EPISODE_TIME_S}s (reset ${RESET_TIME_S}s) at ${FPS}fps"
echo

exec run lerobot-record \
  --config_path="$CONFIG" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.fps="$FPS" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  "${WATCH_ARGS[@]+"${WATCH_ARGS[@]}"}" \
  "$@"
