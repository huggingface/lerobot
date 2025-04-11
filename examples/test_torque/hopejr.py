import threading
import time
from typing import Callable

import cv2
import numpy as np
import serial

from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)

LOWER_BOUND_LINEAR = -100
UPPER_BOUND_LINEAR = 200

ESCAPE_KEY_ID = 27


class HopeJuniorRobot:
    def __init__(self):
        self.arm_bus = FeetechMotorsBus(
            port="/dev/ttyACM1",
            motors={
                # "motor1": (2, "sts3250"),
                # "motor2": (1, "scs0009"),
                #"shoulder_pitch": [1, "sts3250"],
                #"shoulder_yaw": [2, "sts3215"],  # TODO: sts3250
                #"shoulder_roll": [3, "sts3215"],  # TODO: sts3250
                #"elbow_flex": [4, "sts3250"],
                #"wrist_roll": [5, "sts3215"],
                #"wrist_yaw": [6, "sts3215"],
                "wrist_pitch": [7, "sts3215"],
            },
            protocol_version=0,
        )
        self.hand_bus = FeetechMotorsBus(
            port="/dev/ttyACM1",
            motors={
                "thumb_basel_rotation": [30, "scs0009"],
                "thumb_flexor": [27, "scs0009"],
                "thumb_pinky_side": [26, "scs0009"],
                "thumb_thumb_side": [28, "scs0009"],
                "index_flexor": [25, "scs0009"],
                "index_pinky_side": [31, "scs0009"],
                "index_thumb_side": [32, "scs0009"],
                "middle_flexor": [24, "scs0009"],
                "middle_pinky_side": [33, "scs0009"],
                "middle_thumb_side": [34, "scs0009"],
                "ring_flexor": [21, "scs0009"],
                "ring_pinky_side": [36, "scs0009"],
                "ring_thumb_side": [35, "scs0009"],
                "pinky_flexor": [23, "scs0009"],
                "pinky_pinky_side": [38, "scs0009"],
                "pinky_thumb_side": [37, "scs0009"],
            },
            protocol_version=1,
            group_sync_read=False,
        )

    def get_hand_calibration(self):
        """
        Returns a dictionary containing calibration settings for each motor
        on the hand bus.
        """
        homing_offset = [0] * len(self.hand_bus.motor_names)
        drive_mode = [0] * len(self.hand_bus.motor_names)

        start_pos = [
            500, 900, 0, 1000, 100, 250, 750, 100, 400, 150, 100, 120, 980, 100, 950, 750,
        ]

        end_pos = [
            start_pos[0] - 400,    # 500 - 400 = 100
            start_pos[1] - 300,    # 900 - 300 = 600
            start_pos[2] + 550,    #   0 + 550 = 550
            start_pos[3] - 550,    # 1000 - 550 = 450
            start_pos[4] + 900,    # 100 + 900 = 1000
            start_pos[5] + 500,    # 250 + 500 = 750
            start_pos[6] - 500,    # 750 - 500 = 250
            start_pos[7] + 900,    # 100 + 900 = 1000
            start_pos[8] + 700,    # 400 + 700 = 1100
            start_pos[9] + 700,    # 150 + 700 = 850
            start_pos[10] + 900,   # 100 + 900 = 1000
            start_pos[11] + 700,   # 120 + 700 = 820
            start_pos[12] - 700,   # 980 - 700 = 280
            start_pos[13] + 900,   # 100 + 900 = 1000
            start_pos[14] - 700,   # 950 - 700 = 250
            start_pos[15] - 700,   # 750 - 700 = 50
        ]

        calib_modes = [CalibrationMode.LINEAR.name] * len(self.hand_bus.motor_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "calib_mode": calib_modes,
            "motor_names": self.hand_bus.motor_names,
        }
        return calib_dict

    def get_arm_calibration(self):
        """
        Returns a dictionary containing calibration settings for each motor
        on the arm bus.
        """
        homing_offset = [0] * len(self.arm_bus.motor_names)
        drive_mode = [0] * len(self.arm_bus.motor_names)

        # Example placeholders
        start_pos = [
            600,   # shoulder_up
            1500,  # shoulder_forward
            1300,  # shoulder_yaw
            1000,  # bend_elbow
            1600,  # wrist_roll
            1700,  # wrist_yaw
            600,   # wrist_pitch
        ]

        end_pos = [
            2300,  # shoulder_up
            2300,  # shoulder_forward
            2800,  # shoulder_yaw
            2500,  # bend_elbow
            2800,  # wrist_roll
            2200,  # wrist_yaw
            1700,  # wrist_pitch
        ]

        calib_modes = [CalibrationMode.LINEAR.name] * len(self.arm_bus.motor_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "calib_mode": calib_modes,
            "motor_names": self.arm_bus.motor_names,
        }
        return calib_dict

    def connect(self):
        """Connect to the Feetech buses."""
        self.arm_bus.connect()
        # self.hand_bus.connect()


def capture_and_display_processed_frames(
    frame_processor: Callable[[np.ndarray], np.ndarray],
    window_display_name: str,
    cap_device: int = 0,
) -> None:
    """
    Capture frames from the given input camera device, run them through
    the frame processor, and display the outputs in a window with the given name.

    User should press Esc to exit.

    Inputs:
        frame_processor: Callable[[np.ndarray], np.ndarray]
            Processes frames.
            Input and output are numpy arrays of shape (H W C) with BGR channel layout and dtype uint8 / byte.
        window_display_name: str
            Name of the window used to display frames.
        cap_device: int
            Identifier for the camera to use to capture frames.
    """
    cv2.namedWindow(window_display_name)
    capture = cv2.VideoCapture(cap_device)
    if not capture.isOpened():
        raise ValueError("Unable to open video capture.")

    frame_count = 0
    has_frame, frame = capture.read()
    while has_frame:
        frame_count = frame_count + 1
        # Mirror frame horizontally and flip color for demonstration
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])

        # process & show frame
        processed_frame = frame_processor(frame)
        cv2.imshow(window_display_name, processed_frame[:, :, ::-1])

        has_frame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == ESCAPE_KEY_ID:
            break

    capture.release()
