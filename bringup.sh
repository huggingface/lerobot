#!/bin/bash
python -m lerobot.scripts.lerobot_bringup \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AAF2879361 \
    --robot.id=follower \
    "$@"
