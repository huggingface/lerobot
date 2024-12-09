import time
from typing import Callable

import cv2
import numpy as np
from qai_hub_models.models.mediapipe_hand.app import MediaPipeHandApp
from qai_hub_models.models.mediapipe_hand.model import (
    MediaPipeHand,
)
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
)

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus


class HopeJuniorRobot:
    def __init__(self):
        self.arm_bus = FeetechMotorsBus(
            port="/dev/tty.usbserial-2110",
            motors={
                # "motor1": (2, "sts3250"),
                # "motor2": (1, "scs0009"),
                "shoulder_pitch": [1, "sts3250"],
                "shoulder_yaw": [2, "sts3215"],  # TODO: sts3250
                "shoulder_roll": [3, "sts3215"],  # TODO: sts3250
                "elbow_flex": [4, "sts3250"],
                "wrist_roll": [5, "sts3215"],
                "wrist_yaw": [6, "sts3215"],
                "wrist_pitch": [7, "sts3215"],
            },
            protocol_version=0,
        )
        self.hand_bus = FeetechMotorsBus(
            port="/dev/tty.usbserial-2140",
            motors={
                "thumb_basel_rotation": [30, "scs0009"],
                "thumb_flexion": [27, "scs0009"],
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

    def connect(self):
        self.arm_bus.connect()
        self.hand_bus.connect()


ESCAPE_KEY_ID = 27


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
        assert isinstance(frame, np.ndarray)

        frame_count = frame_count + 1
        # mirror frame
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])

        # process & show frame
        processed_frame = frame_processor(frame)
        cv2.imshow(window_display_name, processed_frame[:, :, ::-1])

        has_frame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == ESCAPE_KEY_ID:
            break

    capture.release()


def main():
    robot = HopeJuniorRobot()
    robot.connect()

    print(robot.arm_bus.read("Present_Position"))
    # print(motors_bus.write("Goal_Position", 500))
    print(robot.hand_bus.read("Present_Position"))
    # pos = hand_bus.read("Present_Position")
    # hand_bus.write("Goal_Position", pos[0]+20, hand_bus.motor_names[0])
    # hand_bus.write("Goal_Position", pos[i]+delta, hand_bus.motor_names[i])

    robot.arm_bus.write("Torque_Enable", 1)

    robot.arm_bus.write("Acceleration", 20)
    robot.arm_bus.read("Acceleration")

    robot.hand_bus.read("Acceleration")
    robot.hand_bus.write("Acceleration", 10)

    sleep = 1
    robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
    robot.hand_bus.write(
        "Goal_Position",
        [500, 1000, 0, 1000],
        ["thumb_basel_rotation", "thumb_flexion", "thumb_pinky_side", "thumb_thumb_side"],
    )
    time.sleep(sleep)
    robot.hand_bus.write(
        "Goal_Position", [100, 950, 100], ["index_flexor", "index_pinky_side", "index_thumb_side"]
    )
    time.sleep(sleep)
    robot.hand_bus.write(
        "Goal_Position", [100, 1000, 150], ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
    )
    time.sleep(sleep)
    robot.hand_bus.write(
        "Goal_Position", [200, 200, 0], ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
    )
    time.sleep(sleep)
    robot.hand_bus.write(
        "Goal_Position", [200, 100, 700], ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
    )
    time.sleep(sleep)

    time.sleep(3)

    def move_arm(loop=10):
        sleep = 1
        for i in range(loop):
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1195])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 2195])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1457, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 2357, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 974, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 2674, 1957, 1695])
            time.sleep(sleep + 2)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 1632, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 1369, 1632, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 1330, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [2381, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1681, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)
            robot.arm_bus.write("Goal_Position", [1981, 2030, 2069, 2032, 1874, 1957, 1695])
            time.sleep(sleep)

    def move_hand(loop=10):
        sleep = 0.5
        for i in range(loop):
            robot.hand_bus.write(
                "Goal_Position",
                [500, 1000, 0, 1000],
                ["thumb_basel_rotation", "thumb_flexion", "thumb_pinky_side", "thumb_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [100, 950, 100], ["index_flexor", "index_pinky_side", "index_thumb_side"]
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [100, 1000, 150], ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [200, 200, 0], ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position", [200, 100, 700], ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
            )
            time.sleep(sleep)

            robot.hand_bus.write(
                "Goal_Position",
                [500, 1000 - 250, 0 + 300, 1000 - 200],
                ["thumb_basel_rotation", "thumb_flexion", "thumb_pinky_side", "thumb_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [100 + 450, 950 - 400, 100 + 400],
                ["index_flexor", "index_pinky_side", "index_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [100 + 350, 1000 - 450, 150 + 450],
                ["middle_flexor", "middle_pinky_side", "middle_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [200 + 650, 200 + 350, 0 + 350],
                ["ring_flexor", "ring_pinky_side", "ring_thumb_side"],
            )
            time.sleep(sleep)
            robot.hand_bus.write(
                "Goal_Position",
                [200 + 450, 100 + 400, 700 - 400],
                ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"],
            )
            time.sleep(sleep)

    move_hand(3)

    move_arm(1)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(move_arm)
        executor.submit(move_hand)

    breakpoint()
    # # initial position
    # for i in range(3):
    #     robot.hand_bus.write("Goal_Position", [500, 1000, 0, 1000, 100, 950, 100, 100, 1000, 150, 200, 200, 0, 200, 100, 700])
    #     time.sleep(1)

    # for i in range(3):
    #     robot.hand_bus.write("Goal_Position", [500, 1000-150, 0+250, 1000-150,
    #                                             100+300, 950-250, 100+250,
    #                                             100+200, 1000-300, 150+300,
    #                                             200+500, 200+200, 0+200,
    #                                             200+300, 100+200, 700-200])
    #     time.sleep(1)

    camera = 0
    score_threshold = 0.95
    iou_threshold = 0.3

    app = MediaPipeHandApp(MediaPipeHand.from_pretrained(), score_threshold, iou_threshold)

    def frame_processor(frame: np.ndarray) -> np.ndarray:
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(frame)

        # Run Bounding Box & Keypoint Detector
        batched_selected_boxes, batched_selected_keypoints = app._run_box_detector(NCHW_fp32_torch_frames)

        # The region of interest ( bounding box of 4 (x, y) corners).
        # list[torch.Tensor(shape=[Num Boxes, 4, 2])],
        # where 2 == (x, y)
        #
        # A list element will be None if there is no selected ROI.
        batched_roi_4corners = app._compute_object_roi(batched_selected_boxes, batched_selected_keypoints)

        # selected landmarks for the ROI (if any)
        # list[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
        # where K == number of landmark keypoints, 3 == (x, y, confidence)
        #
        # A list element will be None if there is no ROI.
        landmarks_out = app._run_landmark_detector(NHWC_int_numpy_frames, batched_roi_4corners)

        app._draw_predictions(
            NHWC_int_numpy_frames,
            batched_selected_boxes,
            batched_selected_keypoints,
            batched_roi_4corners,
            *landmarks_out,
        )

        return NHWC_int_numpy_frames[0]

    capture_and_display_processed_frames(frame_processor, "QAIHM Mediapipe Hand Demo", camera)


if __name__ == "__main__":
    main()
