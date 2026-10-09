#!/usr/bin/env bash
# Run ON THE CART (elroy, the Orin). Records a dataset while the operator
# drives from the Pi.
#
#   ./config/cart-record.sh <repo_id> "<task description>"
#   ./config/cart-record.sh --watch carlkesselman/xlerobot-pick "Pick the block, drop it in the box"
#   ./config/cart-record.sh --full --watch me/ds "task"
#
# Flags, in any order, before the repo id:
#   --watch     video to a rerun viewer - see config/_video.sh for
#               --watch=web and --watch=local
#   --arms      arms only instead of the full robot (the default)
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

# --full and --watch in any order. The old code tested $1 for --full, then
# $1 for --watch, so `--watch --full` silently fell through to the ARMS
# config - which on elroy means bi_so_follower, no calibration, and a fresh
# one started over the top of the robot's. An argument order that destroys
# a calibration is not an argument order.
#
# The full robot is also now the DEFAULT. --arms is the opt-in, because the
# dangerous choice should be the one you have to ask for.
CONFIG="$REPO/config/cart-remote.yaml"
ARGS=()
for a in "$@"; do
  case "$a" in
    --full)  CONFIG="$REPO/config/cart-remote.yaml" ;;
    --arms)  CONFIG="$REPO/config/cart-remote-arms.yaml" ;;
    *)       ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
guard_arms_config "$CONFIG"
# `|| :` is load-bearing. parse_watch_all returns 1 when there is no
# --watch flag, and under `set -e` a bare non-zero statement exits the
# script - silently, right after printing the header, which looks
# exactly like a program that started and stopped.
parse_watch_all "$@" || :
set -- "${REMAINING[@]+"${REMAINING[@]}"}"
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
