import argparse
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from examples.real_robot_example.gym_real_world.robot import Robot
import signal
import sys
import pyrealsense2 as rs

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


def capture_frame(camera: cv2.VideoCapture, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, output_color="rgb"):
    # OpenCV acquires frames in BGR format (blue, green red)
    ret, frame = camera.read()
    if not ret:
        raise OSError(f"Camera not found.")

    if output_color == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # Define your crop coordinates (top left corner and bottom right corner)
    # x1, y1 = 400, 0  # Example starting coordinates (top left of the crop rectangle)
    # x2, y2 = 1600, 900  # Example ending coordinates (bottom right of the crop rectangle)
    # # Crop the image
    # image = image[y1:y2, x1:x2]
    # Resize the image
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

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

def find_camera_ids(out_dir="outputs/find_camera_ids/2024_06_19_1039"):
    """
        Install: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

        List cameras and make sure the firmware is up-to-date: https://dev.intelrealsense.com/docs/firmware-releases-d400
        ```bash
        rs-fw-update -l

        > Connected devices:
        > 1) [USB] Intel RealSense D405 s/n 128422270109, update serial number: 133323070634, firmware version: 5.16.0.1
        > 2) [USB] Intel RealSense D405 s/n 128422271609, update serial number: 130523070758, firmware version: 5.16.0.1
        > 3) [USB] Intel RealSense D405 s/n 128422271614, update serial number: 133323070576, firmware version: 5.16.0.1
        > 4) [USB] Intel RealSense D405 s/n 128422271393, update serial number: 133323070271, firmware version: 5.16.0.1
        ```
    """
    save_images = False
    # enable once, if you reach "Frame didn't arrive" exception
    force_hardware_reset = False

    # Works well!
    # use_depth = False
    # fps = 90
    # width = 640
    # height = 480

    # # Works well!
    # use_depth = True
    # fps = 90
    # width = 640
    # height = 480

    # # Doesn't work well, latency varies too much
    # use_depth = True
    # fps = 30
    # width = 1280
    # height = 720

    # Works well
    use_depth = False
    fps = 30
    width = 1280
    height = 720

    out_dir += f"_{width}x{height}_{fps}_depth_{use_depth}"

    ctx = rs.context()

    serials = []
    cameras = []
    for device in ctx.query_devices():
        print(device)
        if force_hardware_reset:
            device.hardware_reset()

        SERIAL_NUMBER_INDEX = 1
        serial_number = device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX))

        config = rs.config()
        config.enable_device(serial_number)

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        if use_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        pipeline = rs.pipeline()
        pipeline.start(config)

        serials.append(serial_number)
        cameras.append(pipeline)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            now = time.time()
            for serial, camera in zip(serials, cameras):
                # Wait for a coherent pair of frames: depth and color
                try:
                    frames = camera.wait_for_frames()
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        print(f"{e}: Trying hardware_reset. If it still doesn't work, try `force_hardware_reset=True`.")
                        device.hardware_reset()
                    traceback.print_exc()
                    continue

                # acquire color image
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("Empty color frame")
                    continue
                # to numpy
                image = np.asanyarray(color_frame.get_data())

                if save_images:
                    image_path = out_dir / f"camera_{serial:02}.png"
                    print(f"Write to {image_path}")
                    write_shape(image)
                    cv2.imwrite(str(image_path), image)

                if use_depth:
                    # acquire depth image
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        print("Empty depth frame")
                        continue
                    # to numpy
                    depth = np.asanyarray(depth_frame.get_data())
                    
                    if save_images:
                        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                        depth_image_path = out_dir / f"camera_{serial:02}_depth.png"
                        print(f"Write to {depth_image_path}")
                        write_shape(depth_image)
                        cv2.imwrite(str(depth_image_path), depth_image)

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
            camera.stop()

    return serials


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

        print(f"Time to send pos: {(time.time() - now) * 1000}")

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
