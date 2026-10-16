#!/usr/bin/env bash
# Run ON THE CART (elroy, the Orin). Records a dataset while the operator
# drives from the Pi.
#
#   ./config/cart-record.sh <repo_id> "<task description>"
#   ./config/cart-record.sh --watch carlkesselman/xlerobot-pick "Pick the block, drop it in the box"
#   BASE=none ./config/cart-record.sh me/ds "task"      arms and head only
#
# Flags, in any order, before the repo id:
#   --watch     video to a rerun viewer - see config/_video.sh for
#               --watch=web and --watch=local
#   BASE=       none | omniwheels | odrive - which base the dataset will
#               have; its features follow (no .vel columns with none)
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
# DEFAULTS THIS SCRIPT SETS (each overridable by passing it again)
#   --dataset.push_to_hub=false        local only. PUSH_TO_HUB=true to upload
#                                      (hf auth login once, first).
#   --dataset.rgb_encoder.vcodec=h264  VCODEC=. lerobot's default is AV1
#                                      (libsvtav1), several times slower to
#                                      encode on the Orin's CPU; h264 keeps the
#                                      between-episode encode inside reset_time.
#   --dataset.num_image_writer_processes=2
#                                      frame PNGs written in 2 processes, not
#                                      12 threads competing with the control
#                                      loop for the GIL.
#   --play_sounds=false                lerobot speaks through spd-say, which a
#                                      headless Orin may not have; a missing
#                                      binary ends the run before tick one.
#
# The dataset lands in ~/.cache/huggingface/lerobot/<repo_id>_YYYYMMDD_HHMMSS:
# lerobot stamps every new dataset with the date and time
# (--dataset.no_stamp=true to stop it). The path is printed when the run
# ends; ./config/dataset-check.sh <repo_id> reads back the newest.
#
# ENDING: right arrow ends an episode and saves it, left re-records it,
# Escape ends AND SAVES the current one, then stops. Ctrl-C loses the
# episode in progress.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"

warn_if_cpu_torch

CONFIG="$REPO/config/cart-remote.yaml"
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

# A flag in the repo-id slot - a typo, or one this script does not parse -
# would otherwise become the dataset's name.
if [[ "$REPO_ID" == -* || "$REPO_ID" != */* ]]; then
  echo "error: repo id must look like <user>/<name>, got '$REPO_ID'" >&2
  echo "       (flags go after the task, except --watch and BASE=)" >&2
  exit 2
fi

# The yaml is shared with cart-teleop.sh, and its top-level `fps:` is a
# TeleoperateConfig field. RecordConfig has none - its rate is dataset.fps -
# and draccus rejects unknown fields, so record from a copy without it.
RECORD_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$RECORD_CONFIG"' EXIT
grep -v '^fps:' "$CONFIG" > "$RECORD_CONFIG"

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
base_report "$CONFIG"
echo "leaders : $OPERATOR:5557"
echo "dataset : $REPO_ID"
echo "task    : $TASK"
echo "episodes: $NUM_EPISODES x ${EPISODE_TIME_S}s (reset ${RESET_TIME_S}s) at ${FPS}fps"
echo "upload  : ${PUSH_TO_HUB:-false}   codec: ${VCODEC:-h264}"
DATA_ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
echo "saved to: $DATA_ROOT/${REPO_ID}_<date>_<time>"
echo

# Not exec: the trap above has to run to remove the temporary yaml, and the
# dataset's stamped path is reported afterwards. `|| rc=$?` because _common.sh
# sets -e, which would otherwise end the script on a non-zero exit here.
rc=0
"${RUN[@]}" lerobot-record \
  --config_path="$RECORD_CONFIG" \
  "${BASE_ARGS[@]+"${BASE_ARGS[@]}"}" \
  --dataset.push_to_hub="${PUSH_TO_HUB:-false}" \
  --dataset.rgb_encoder.vcodec="${VCODEC:-h264}" \
  --dataset.num_image_writer_processes=2 \
  --play_sounds=false \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.fps="$FPS" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  "${DISPLAY_ARGS[@]+"${DISPLAY_ARGS[@]}"}" \
  "$@" || rc=$?

newest="$(ls -dt "$DATA_ROOT/${REPO_ID}"_* 2>/dev/null | head -1 || true)"
echo
if [[ -n "$newest" ]]; then
  echo "dataset : $newest"
  echo "check it: ./config/dataset-check.sh $REPO_ID"
else
  echo "no dataset found under $DATA_ROOT/${REPO_ID}_*"
fi
exit $rc
