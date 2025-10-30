#!/bin/bash

# List of device strings
device_list=(
  "/dev/ttyACM0"
  "/dev/ttyACM1"
  "/dev/ttyACM2"
  "/dev/ttyACM3"
  "/dev/ttyACM4"
)

# Loop through each device in the list
for stringx in "${device_list[@]}"; do
  # Check for "follower" identifier (5AAF219986)
  result_follower=$(udevadm info "$stringx" | grep "5AAF219986")

  # Check for "leader" identifier (5AAF218179)
  result_leader=$(udevadm info "$stringx" | grep "5AAF218179")

  # Determine if it's a follower or leader based on the result
  if [[ -n "$result_follower" ]]; then
    echo "$stringx : follower!!!!"
  elif [[ -n "$result_leader" ]]; then
    echo "$stringx : leader!!!!"

  fi
done