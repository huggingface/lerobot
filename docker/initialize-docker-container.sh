#!/bin/bash
# install octo
source ~/.bashrc
cd ~/lerobot && poetry install --extras "aloha xarm pusht"

# git safe directory
cd ~/lerobot && git config --global --add safe.directory /root/lerobot

cd ~/lerobot

# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null

