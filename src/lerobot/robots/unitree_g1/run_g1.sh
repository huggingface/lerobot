#!/usr/bin/env bash
# Launch the G1 ZMQ bridge server with grippers + cameras.
# Handles conda activation and CAN bring-up so you only run one command on the robot.
set -euo pipefail

# --- conda -------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "conda.sh not found at $CONDA_SH (set CONDA_SH=/path/to/conda.sh)" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate lerobot

# --- CAN bring-up + test -----------------------------------------------------
CAN_INTERFACES="${CAN_INTERFACES:-can0,can1}"
echo "==> Setting up CAN interfaces: $CAN_INTERFACES"
lerobot-setup-can --mode=setup --interfaces="$CAN_INTERFACES"

echo "==> Testing CAN motors on: $CAN_INTERFACES"
lerobot-setup-can --mode=test --interfaces="$CAN_INTERFACES"

# --- G1 server ---------------------------------------------------------------
CAMERAS="${CAMERAS:-head_camera:/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_254343063964-video-index0:640x480:YUYV,left_wrist:/dev/video2:1280x720:MJPG,right_wrist:/dev/video0:1280x720:MJPG}"
CAMERA_FPS="${CAMERA_FPS:-30}"
GRIPPER_PORT_LEFT="${GRIPPER_PORT_LEFT:-can1}"
GRIPPER_PORT_RIGHT="${GRIPPER_PORT_RIGHT:-can0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Starting G1 server"
exec python "$SCRIPT_DIR/run_g1_server.py" \
    --grippers \
    --gripper-port-left "$GRIPPER_PORT_LEFT" \
    --gripper-port-right "$GRIPPER_PORT_RIGHT" \
    --camera-fps "$CAMERA_FPS" \
    --cameras "$CAMERAS"
