lerobot-teleoperate \
    --robot.type=xlerobot \
    --robot.base_type=lekiwi_base \
    --robot.arms='{
        "left_arm_port": "/dev/ttyACM2",
        "right_arm_port": "/dev/ttyACM3",
        "id": "follower"
    }' \
    --robot.base='{
        "port": "/dev/ttyACM4",
        "wheel_radius_m": 0.05,
        "base_radius_m": 0.125
    }' \
    --robot.mount='{
        "port": "/dev/ttyACM5",
        "pan_motor_id": 0,
        "tilt_motor_id": 1,
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
    --teleop.type=xlerobot_leader_gamepad \
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
#     --robot.base_type=lekiwi_base \
#     --robot.arms='{}' \
#     --robot.base='{
#         "port": "/dev/ttyACM4",
#         "wheel_radius_m": 0.05,
#         "base_radius_m": 0.125
#     }' \
#     --robot.mount='{
#         "port": "/dev/ttyACM5",
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
#     --teleop.type=xlerobot_leader_gamepad \
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
