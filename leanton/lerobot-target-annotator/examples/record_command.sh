#!/usr/bin/env bash
# lerobot-record command for use with lerobot-target-annotator.
#
# Prerequisites:
#   - annotate_stream.py must be running in a separate terminal (Terminal 1)
#   - conda activate lerobot
#
# Usage:
#   bash examples/record_command.sh
#
# Edit the variables below to match your setup.

# ── Configuration ─────────────────────────────────────────────
FOLLOWER_PORT="<FOLLOWER_PORT>"       # e.g. /dev/tty.usbmodem5B420769691 (macOS) or /dev/so101_follower (Linux)
FOLLOWER_ID="<FOLLOWER_ROBOT_ID>"   # e.g. robotnik_follower_arm

LEADER_PORT="<LEADER_PORT>"         # e.g. /dev/tty.usbmodem5B420771281 (macOS) or /dev/so101_leader (Linux)
LEADER_ID="<LEADER_ROBOT_ID>"       # e.g. robotnik_leader_arm

HF_REPO="anikitakis/pick_and_place_visual_prompt"

NUM_EPISODES=50
EPISODE_TIME_S=20
RESET_TIME_S=10
TASK="pick the object and place it in the basket"

ZMQ_PORT=5555
# ──────────────────────────────────────────────────────────────

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="{
    \"front\":     {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30, \"fourcc\": \"MJPG\"},
    \"annotated\": {\"type\": \"zmq\",    \"server_address\": \"localhost\", \"port\": ${ZMQ_PORT}, \"camera_name\": \"annotated\", \"width\": 640, \"height\": 480, \"fps\": 30}
  }" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}" \
  --display_data=true \
  --dataset.repo_id="${HF_REPO}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK}" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}"
