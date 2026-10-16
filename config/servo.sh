#!/usr/bin/env bash
# The cart's servo tools. Run ON THE CART.
#
#   ./config/servo.sh scan              which ids answer, at which baud
#   ./config/servo.sh power             voltage under progressive load
#   ./config/servo.sh regs [--bus 1]    every register, flags the odd one out
#   ./config/servo.sh set-id --bus 2 --id 5 [--port ...]   one new servo
#   ./config/servo.sh recal right_arm_wrist_roll           one joint
#
# Ports and joint ids live in config/servo_tools/hw.py, once.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export PYTHONPATH="$REPO/config${PYTHONPATH:+:$PYTHONPATH}"
exec "${RUN[@]}" python -m servo_tools "$@"
