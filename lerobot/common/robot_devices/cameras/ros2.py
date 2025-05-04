"""
This file contains utilities for recording frames from cameras. For more info look at `ROS2Camera` docstring.
"""

import argparse
import concurrent.futures
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

# TODO(Yadunund): Implement mock.
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg

from lerobot.common.robot_devices.cameras.configs import ROS2CameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc


def save_image(img_array, camera_name, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_name:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path,
    topic: str | None = None,
    record_time_s=2,
    mock=False,
):
    print("Connecting cameras")
    cameras = []
    config = ROS2CameraConfig(topic=topic, mock=mock)
    camera = ROS2Camera(config)
    camera.connect()
    print(f"ROS2Camera({camera.node.get_name()}")
    cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read()
                executor.submit(
                    save_image,
                    image,
                    config.topic,
                    frame_index,
                    images_dir,
                )

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    print(f"Images have been saved to {images_dir}")


class ROS2Camera:
    """
    The ROS2Camera class records images from cameras via ROS 2.

    This class requires a ROS 2 camera driver to be running and publishing images on a specified topic.
    It's compatible with any camera that has a ROS 2 driver, whether running on the same host or another host on the network.

    To discover available camera topics:
    ```bash
    # Source your ROS 2 installation first
    ros2 topic list
    ```

    Basic usage:
    ```python
    from lerobot.common.robot_devices.cameras.configs import ROS2CameraConfig

    config = ROS2CameraConfig(topic="/camera/image")
    camera = ROS2Camera(config)
    camera.connect()
    image = camera.read()
    camera.disconnect()  # Clean up when done
    ```

    Configuration notes:
    - The topic name is required for connection
    - Other parameters (fps, width, height) should match the actual ROS 2 driver settings
    - Width and height will be updated from the first received message
    - Rotation can be applied (90, -90, 180 degrees) if specified in the config

    ```python
    config = ROS2CameraConfig(
        topic="/camera/image_raw",
        fps=30,
        width=640,
        height=480,
        rotation=90  # Optional rotation
    )
    ```
    """

    def __init__(self, config: ROS2CameraConfig):
        rclpy.init()
        self.config = config
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock

        self.br: CvBridge = CvBridge()
        self.image_msg: ImageMsg | None = None
        self.is_connected: bool = False
        self.sub = None
        self.logs = {}
        self.node = rclpy.create_node("lerobot_camera_node")
        self.stop_event = threading.Event()
        self.spin_thread = Thread(target=self.spin_node)
        self.spin_thread.start()

        # TODO(aliberts): Do we keep original width/height or do we define them after rotation?
        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def spin_node(self):
        while rclpy.ok() and not self.stop_event.is_set():
            rclpy.spin_once(self.node)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"ROS2Camera({self.config.topic}) is already connected.")

        self.sub = self.node.create_subscription(ImageMsg, self.config.topic, self.sub_cb, 10)

        while self.image_msg is None:
            print(f"Waiting to receive message over {self.config.topic}")
            time.sleep(1)

        print(f"Successfully connected to ROS2Camera({self.config.topic})!")
        self.is_connected = True

    def sub_cb(self, msg):
        self.image_msg = msg
        self.width = msg.width
        self.height = msg.height

    def read(self, temporary_color: str | None = None) -> np.ndarray:
        """Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected or self.image_msg is None:
            raise RobotDeviceNotConnectedError(
                f"ROS2Camera({self.node.get_name()}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        desired_encoding = "passthrough" if temporary_color is None else temporary_color

        image = self.br.imgmsg_to_cv2(self.image_msg, desired_encoding)

        if self.rotation is not None:
            image = cv2.rotate(image, self.rotation)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        return image

    def async_read(self):
        return self.read()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ROS2Camera({self.node.get_name()}) is not connected. Try running `camera.connect()` first."
            )

        if self.spin_thread is not None:
            self.stop_event.set()
            self.spin_thread.join()  # wait for the thread to finish
            self.spin_thread = None
            self.stop_event = None

        self.sub = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `ROS2Camera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--topic",
        type=str,
        nargs="*",
        default=None,
        help="Name of the topic to subscribe to for images.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
