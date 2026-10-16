#!/usr/bin/env bash
# Recalibrate one joint after a servo swap; every other joint keeps its
# calibration. Backs up the calibration file first. See recal_joint.py.
#
#   ./config/recal-joint.sh right_arm_wrist_roll
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUN[@]}" python "$REPO/config/recal_joint.py" "$@"
