#! /bin/bash

#/bin/bash

# python lerobot/scripts/control_robot.py --config_path lerobot/configs/robot/koch_jack.yaml --control.type teleoperate --control.fps 30

# old config for reference
# robot:
#   type: koch
#   calibration_dir: .cache/calibration/koch_tdmpc_jack
#   leader_arms:
#     main:
#       type: dynamixel
#       port: /dev/servo_585A007782
#       motors:
#         # name: (index, model)
#         shoulder_pan: [1, "xl330-m077"]
#         shoulder_lift: [2, "xl330-m077"]
#         elbow_flex: [3, "xl330-m077"]
#         wrist_flex: [4, "xl330-m077"]
#         wrist_roll: [5, "xl330-m077"]
#         gripper: [6, "xl330-m077"]
#   follower_arms:
#     main:
#       type: dynamixel
#       port: /dev/servo_5837053138
#       motors:
#         # name: (index, model)
#         shoulder_pan: [1, "xl430-w250"]
#         shoulder_lift: [2, "xl430-w250"]
#         elbow_flex: [3, "xl330-m288"]
#         wrist_flex: [4, "xl330-m288"]
#         wrist_roll: [5, "xl330-m288"]
#         gripper: [6, "xl330-m288"]
#   cameras:
#     # main:
#     #   _target_: lerobot.common.robot_devices.cameras.opencv.OpenCVCamera
#     #   camera_index: 0
#     #   fps: 15
#     #   width: 800
#     #   height: 600
#     top:
#       type: opencv
#       camera_index: 5
#       fps: 15
#       width: 800
#       height: 600
#     # side:
#     #   type: opencv
#     #   camera_index: 4
#     #   fps: 15
#     #   width: 800
#     #   height: 600
#     # top2:
#     #   _target_: lerobot.common.robot_devices.cameras.opencv.OpenCVCamera
#     #   camera_index: 8
#     #   fps: 15
#     #   width: 800
#     #   height: 600
#   gripper_open_degree: 35.156
#   # Clamp the magnitude (in degrees) any joint is allowed to move in a single
#   # action step.  Either a scalar applied to all joints or a list with one
#   # entry per joint.  Here we use a conservative scalar.
#   # max_relative_target: 10  # degrees per control step
#   gripper_mode: "screwdriver"

# control:
#   # If you need to specify which arms to calibrate, you can add an 'arms' field here.
#   # For example, to calibrate only the 'main' arm:
#   # arms: ["main"]

# Cameras

# --- Detected Cameras ---
# Camera #0:
#   Name: OpenCV Camera @ /dev/video0
#   Type: OpenCV
#   Id: /dev/video0
#   Backend api: V4L2
#   Default stream profile:
#     Format: 0.0
#     Width: 640
#     Height: 480
#     Fps: 30.0
# --------------------
# Camera #1:
#   Name: OpenCV Camera @ /dev/video2
#   Type: OpenCV
#   Id: /dev/video2
#   Backend api: V4L2
#   Default stream profile:
#     Format: 0.0
#     Width: 640
#     Height: 480
#     Fps: 30.0
# --------------------
# Camera #2:
#   Name: OpenCV Camera @ /dev/video6
#   Type: OpenCV
#   Id: /dev/video6
#   Backend api: V4L2
#   Default stream profile:
#     Format: 0.0
#     Width: 640
#     Height: 480
#     Fps: 30.0
# --------------------

# python -m lerobot.teleoperate \
#     --robot.type=koch_screwdriver_follower \
#     --robot.port=/dev/servo_5837053138 \
#     --robot.cameras="{ screwdriver: {type: opencv, index_or_path: /dev/video0, width: 800, height: 600, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, width: 800, height: 600, fps: 30}}" \
#     --robot.id=koch_screwdriver_follower_testing \
#     --teleop.type=koch_screwdriver_leader \
#     --teleop.port=/dev/servo_585A007782 \
#     --teleop.id=koch_screwdriver_leader_testing \
#     --display_data=true

python -m lerobot.record \
    --robot.type=koch_screwdriver_follower \
    --robot.port=/dev/servo_5837053138 \
    --robot.cameras="{ screwdriver: {type: opencv, index_or_path: /dev/video0, width: 800, height: 600, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, width: 800, height: 600, fps: 30}}" \
    --robot.id=koch_screwdriver_follower_testing \
    --dataset.repo_id=jackvial/koch_screwdriver_follower_test_01 \
    --dataset.num_episodes=2 \
    --dataset.single_task="Screw the silver screw into the M4x0.7 thread on the red thread checker board" \
    --teleop.type=koch_screwdriver_leader \
    --teleop.port=/dev/servo_585A007782 \
    --teleop.id=koch_screwdriver_leader_testing \
    --display_data=true