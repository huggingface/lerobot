import argparse
from pathlib import Path
import time
import warnings

import cv2
import numpy as np
from examples.real_robot_example.gym_real_world.robot import Robot
import signal
import sys
# import pyrealsense2 as rs

MAX_LEADER_GRIPPER_RAD = 0.7761942786701344
MAX_LEADER_GRIPPER_POS = 2567
MAX_FOLLOWER_GRIPPER_RAD = 1.6827769243105486
MAX_FOLLOWER_GRIPPER_POS = 3100

MIN_LEADER_GRIPPER_RAD = -0.12732040539450828
MIN_LEADER_GRIPPER_POS = 1984
MIN_FOLLOWER_GRIPPER_RAD = 0.6933593161243099
MIN_FOLLOWER_GRIPPER_POS = 2512

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def convert_gripper_range_from_leader_to_follower(leader_gripper_pos):
    follower_gripper_pos = \
        (leader_gripper_pos - MIN_LEADER_GRIPPER_POS) \
        / (MAX_LEADER_GRIPPER_POS - MIN_LEADER_GRIPPER_POS) \
        * (MAX_FOLLOWER_GRIPPER_POS - MIN_FOLLOWER_GRIPPER_POS) \
        + MIN_FOLLOWER_GRIPPER_POS
    return follower_gripper_pos

# alexander koch
# leader_port = "/dev/ttyACM1"
# follower_port = "/dev/ttyACM0"

def disable_torque():
    leader_right_port = "/dev/ttyDXL_master_right"
    leader_left_port = "/dev/ttyDXL_master_left"
    follower_right_port = "/dev/ttyDXL_puppet_right"
    follower_left_port = "/dev/ttyDXL_puppet_left"
    # starts at 1
    all_servo_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    leader_right = Robot(leader_right_port, servo_ids=all_servo_ids)
    leader_left = Robot(leader_left_port, servo_ids=all_servo_ids)
    follower_right = Robot(follower_right_port, servo_ids=all_servo_ids)
    follower_left = Robot(follower_left_port, servo_ids=all_servo_ids)

    leader_right._disable_torque()
    leader_left._disable_torque()
    follower_right._disable_torque()
    follower_left._disable_torque()


def teleoperate():
    leader_right_port = "/dev/ttyDXL_master_right"
    follower_right_port = "/dev/ttyDXL_puppet_right"
    leader_left_port = "/dev/ttyDXL_master_left"
    follower_left_port = "/dev/ttyDXL_puppet_left"
    # starts at 1
    all_servo_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    leader_right = Robot(leader_right_port, servo_ids=all_servo_ids)
    leader_left = Robot(leader_left_port, servo_ids=all_servo_ids)
    follower_right = Robot(follower_right_port, servo_ids=all_servo_ids)
    follower_left = Robot(follower_left_port, servo_ids=all_servo_ids)

    follower_right._enable_torque()
    follower_left._enable_torque()

    while True:
        now = time.time()
        # Prepare to assign the positions of the leader to the follower 
        follower_right_pos = leader_right.read_position()
        follower_left_pos = leader_left.read_position()

        # Update the position of the follower gripper to account for the different minimum and maximum range
        # position in range [0, 4096[ which corresponds to 4096 bins of 360 degrees
        # for all our dynamixel servors
        # gripper id=8 has a different range from leader to follower
        follower_right_pos[-1] = convert_gripper_range_from_leader_to_follower(follower_right_pos[-1])
        follower_left_pos[-1] = convert_gripper_range_from_leader_to_follower(follower_left_pos[-1])

        # Assign
        follower_right.set_goal_pos(follower_right_pos)
        follower_left.set_goal_pos(follower_left_pos)

        print(f"Time to send pos: {(time.time() - now) * 1000}")


def capture_frame(camera: cv2.VideoCapture, output_color="rgb"):
    # OpenCV acquires frames in BGR format (blue, green red)
    ret, frame = camera.read()
    if not ret:
        raise OSError(f"Camera not found.")

    if output_color == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame

def find_camera_ids(out_dir="outputs/find_camera_ids/2024_06_19_cv2_1344"):
    save_images = True
    max_index_search_range = 60
    num_warmup_frames = 4

    # Works well
    codec = "yuyv"
    fps = 30
    width = 640
    height = 480

    # # Works well
    # codec = "yuyv"
    # fps = 60
    # width = 640
    # height = 480

    # # Works well
    # codec = "yuyv"
    # fps = 90
    # width = 640
    # height = 480

    # # Works well
    # codec = "yuyv"
    # fps = 30
    # width = 1280
    # height = 720

    # Doesn't work well (timeout)
    # codec = "mjpg"
    # fps = 30
    # width = 1280
    # height = 720

    out_dir += f"_{width}x{height}_{fps}_{codec}"

    camera_ids = []
    for camera_idx in range(max_index_search_range):
        camera = cv2.VideoCapture(camera_idx)
        is_open = camera.isOpened()
        camera.release()

        if is_open:
            print(f"Camera found at index {camera_idx}")
            camera_ids.append(camera_idx)

    if len(camera_ids) == 0:
        raise OSError("No camera has been found")
    
    # Change camera settings
    cameras = []
    for camera_idx in camera_ids:
        camera = cv2.VideoCapture(camera_idx)

        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if codec == "mjpg":
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if fps != actual_fps:
            warnings.warn(f"{fps=} != {actual_fps=}", stacklevel=1)
        if width != actual_width:
            warnings.warn(f"{width=} != {actual_width=}", stacklevel=1)
        if height != actual_height:
            warnings.warn(f"{height=} != {actual_height=}", stacklevel=1)

        cameras.append(camera)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Capturing a few frames to warmup")
    for _ in range(num_warmup_frames):
        for camera_idx, camera in zip(camera_ids, cameras):
            print(f"Capturing camera {camera_idx}")
            try:
                frame = capture_frame(camera, output_color="bgr" if save_images else "rgb")
                time.sleep(0.01)
            except OSError as e:
                print(e)
        time.sleep(0.1)

    print("Capturing frames")
    try:
        while True:
            now = time.time()
            for camera_idx, camera in zip(camera_ids, cameras):
                try:
                    frame = capture_frame(camera, output_color="bgr" if save_images else "rgb")
                except OSError as e:
                    print(e)    

                def write_shape(frame):
                    height, width = frame.shape[:2]
                    text = f'Width: {width} Height: {height}'

                    # Define the font, scale, color, and thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 0, 0)  # Blue in BGR
                    thickness = 2

                    position = (10, height - 10)  # 10 pixels from the bottom-left corner
                    cv2.putText(frame, text, position, font, font_scale, color, thickness)

                if save_images:
                    frame_path = out_dir / f"camera_{camera_idx:02}.png"
                    print(f"Write to {frame_path}")
                    write_shape(frame)
                    cv2.imwrite(str(frame_path), frame)
                    time.sleep(0.1)

            dt_s = (time.time() - now)
            dt_ms = dt_s * 1000
            freq = 1 / dt_s
            print(f"Latency (ms): {dt_ms:.2f}\tFrequency: {freq:.2f}")

            if save_images:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Stop streaming
        for camera in cameras:
            camera.release()

    return camera_ids

def record_data():
    leader_right_port = "/dev/ttyDXL_master_right"
    follower_right_port = "/dev/ttyDXL_puppet_right"
    leader_left_port = "/dev/ttyDXL_master_left"
    follower_left_port = "/dev/ttyDXL_puppet_left"
    # starts at 1
    all_servo_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    leader_right = Robot(leader_right_port, servo_ids=all_servo_ids)
    leader_left = Robot(leader_left_port, servo_ids=all_servo_ids)
    follower_right = Robot(follower_right_port, servo_ids=all_servo_ids)
    follower_left = Robot(follower_left_port, servo_ids=all_servo_ids)

    follower_right._enable_torque()
    follower_left._enable_torque()

    # To get the camera_ids, run: `find_camera_ids()`
    camera_high = cv2.VideoCapture(10)
    camera_low = cv2.VideoCapture(22)
    camera_right_wrist = cv2.VideoCapture(16)
    camera_left_wrist = cv2.VideoCapture(4)

    if not camera_high.isOpened():
        raise OSError("Camera high port can't be accessed.")
    if not camera_low.isOpened():
        raise OSError("Camera low port can't be accessed.")
    if not camera_right_wrist.isOpened():
        raise OSError("Camera right_wrist port can't be accessed.")
    if not camera_left_wrist.isOpened():
        raise OSError("Camera left_wrist port can't be accessed.")

    while True:
        now = time.time()

        frame_high = capture_frame(camera_high)
        frame_low = capture_frame(camera_low)
        frame_right_wrist = capture_frame(camera_right_wrist)
        frame_left_wrist = capture_frame(camera_left_wrist)

        # cv2.imshow("high", frame_high)
        # cv2.imshow("low", frame_low)
        # cv2.imshow("right_wrist", frame_right_wrist)
        # cv2.imshow("left_wrist", frame_left_wrist)

        # Prepare to assign the positions of the leader to the follower 
        follower_right_pos = leader_right.read_position()
        follower_left_pos = leader_left.read_position()

        # Update the position of the follower gripper to account for the different minimum and maximum range
        # position in range [0, 4096[ which corresponds to 4096 bins of 360 degrees
        # for all our dynamixel servors
        # gripper id=8 has a different range from leader to follower
        follower_right_pos[-1] = convert_gripper_range_from_leader_to_follower(follower_right_pos[-1])
        follower_left_pos[-1] = convert_gripper_range_from_leader_to_follower(follower_left_pos[-1])

        # Assign
        follower_right.set_goal_pos(follower_right_pos)
        follower_left.set_goal_pos(follower_left_pos)

        dt_s = (time.time() - now)
        dt_ms = dt_s * 1000
        freq = 1 / dt_s
        print(f"Latency (ms): {dt_ms:.2f}\tFrequency: {freq:.2f}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera_high.release()
    camera_low.release()
    camera_right_wrist.release()
    camera_left_wrist.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["teleoperate", "disable_torque", "record_data", "find_camera_ids"], default="teleoperate")
    args = parser.parse_args()

    if args.mode == "teleoperate":
        teleoperate()
    elif args.mode == "disable_torque":
        disable_torque()
    elif args.mode == "record_data":
        record_data()
    elif args.mode == "find_camera_ids":
        find_camera_ids()
