# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# XLeRobot integration based on
#   https://www.hackster.io/brainbot/brainbot-big-brain-with-xlerobot-ad1b4c
#   https://github.com/Astera-org/brainbot
#   https://github.com/Vector-Wangel/XLeRobot
#   https://github.com/bingogome/lerobot

# Demo video
# https://drive.google.com/file/d/1Kqvb8zP6Zjkz2CuB5h4jL4ymOBka8ckQ/view?usp=sharing

lerobot-teleoperate \
    --robot.type=xlerobot \
    --robot.left_arm='{
        "id": "xlerobot_arm_left"
    }' \
    --robot.right_arm='{
        "id": "xlerobot_arm_right"
    }' \
    --robot.base='{
        "type": "lekiwi_base",
        "wheel_radius_m": 0.05,
        "base_radius_m": 0.125
    }' \
    --robot.mount='{
        "pan_motor_id": 1,
        "tilt_motor_id": 2,
        "motor_model": "sts3215",
        "pan_key": "mount_pan.pos",
        "tilt_key": "mount_tilt.pos",
        "max_pan_speed_dps": 60.0,
        "max_tilt_speed_dps": 45.0,
        "pan_range": [-90.0, 90.0],
        "tilt_range": [-30.0, 60.0]
    }' \
    --robot.cameras='{
        "top":   {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30}
    }' \
    --robot.shared_buses='{
        "left_bus": {
            "port": "/dev/ttyACM2",
            "components": [
                {"component": "left_arm"},
                {"component": "mount", "motor_id_offset": 6}
            ]
        },
        "right_bus": {
            "port": "/dev/ttyACM3",
            "components": [
                {"component": "right_arm"},
                {"component": "base", "motor_id_offset": 6}
            ]
        }
    }' \
    --teleop.type=xlerobot_default_composite \
    --teleop.base_type=lekiwi_base_gamepad \
    --teleop.arms='{
        "left_arm_port": "/dev/ttyACM0",
        "right_arm_port": "/dev/ttyACM1",
        "id": "leader"
    }' \
    --teleop.base='{
        "joystick_index": 0,
        "max_speed_mps": 0.8,
        "deadzone": 0.15,
        "yaw_speed_deg": 45
    }' \
    --teleop.mount='{
        "joystick_index": 0,
        "deadzone": 0.15,
        "polling_fps": 50,
        "max_pan_speed_dps": 60.0,
        "max_tilt_speed_dps": 45.0,
        "pan_axis": 3,
        "tilt_axis": 4,
        "invert_pan": false,
        "invert_tilt": true,
        "pan_range": [-90.0, 90.0],
        "tilt_range": [-30.0, 60.0]
    }' \
    --display_data=true


# # Or, if you want to run without a sub-robot, say the arms:
#
# lerobot-teleoperate \
#     --robot.type=xlerobot \
#     --robot.left_arm='{}' \
#     --robot.right_arm='{}' \
#     --robot.base='{
#         "type": "lekiwi_base",
#         "wheel_radius_m": 0.05,
#         "base_radius_m": 0.125
#     }' \
#     --robot.mount='{
#         "pan_motor_id": 0,
#         "tilt_motor_id": 1,
#         "motor_model": "sts3215",
#         "pan_key": "mount_pan.pos",
#         "tilt_key": "mount_tilt.pos",
#         "max_pan_speed_dps": 60.0,
#         "max_tilt_speed_dps": 45.0,
#         "pan_range": [-90.0, 90.0],
#         "tilt_range": [-30.0, 60.0]
#     }' \
#     --robot.cameras='{}' \
#     --robot.shared_buses='{
#         "base_bus": {
#             "port": "/dev/ttyACM4",
#             "components": [
#                 {"component": "base"}
#             ]
#         },
#         "mount_bus": {
#             "port": "/dev/ttyACM5",
#             "components": [
#                 {"component": "mount"}
#             ]
#         }
#     }' \
#     --teleop.type=xlerobot_default_composite \
#     --teleop.base_type=lekiwi_base_gamepad \
#     --teleop.arms='{}' \
#     --teleop.base='{
#         "joystick_index": 0,
#         "max_speed_mps": 0.8,
#         "deadzone": 0.15,
#         "yaw_speed_deg": 45
#     }' \
#     --teleop.mount='{
#         "joystick_index": 0,
#         "deadzone": 0.15,
#         "polling_fps": 50,
#         "max_pan_speed_dps": 60.0,
#         "max_tilt_speed_dps": 45.0,
#         "pan_axis": 3,
#         "tilt_axis": 4,
#         "invert_pan": false,
#         "invert_tilt": true,
#         "pan_range": [-90.0, 90.0],
#         "tilt_range": [-30.0, 60.0]
#     }' \
#     --display_data=true
