import argparse
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread

import cv2
# Using 1 thread to avoid blocking the main thread.
# Especially useful during data collection when other threads are used
# to save the images.
cv2.setNumThreads(1)
import numpy as np

from lerobot.common.robot_devices.cameras.utils import save_color_image
from lerobot.common.utils.utils import capture_timestamp_utc


def find_camera_indices(raise_when_empty=False, max_index_search_range=60):
    camera_ids = []
    for camera_idx in range(max_index_search_range):
        camera = cv2.VideoCapture(camera_idx)
        is_open = camera.isOpened()
        camera.release()

        if is_open:
            print(f"Camera found at index {camera_idx}")
            camera_ids.append(camera_idx)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, or your camera driver, or make sure your camera is compatible with opencv2."
        )

    return camera_ids


def benchmark_cameras(cameras, out_dir=None, save_images=False, num_warmup_frames=4):
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(num_warmup_frames):
        for camera in cameras:
            try:
                camera.capture_image()
                time.sleep(0.01)
            except OSError as e:
                print(e)

    while True:
        now = time.time()
        for camera in cameras:
            color_image = camera.capture_image("bgr" if save_images else "rgb")

            if save_images:
                image_path = out_dir / f"camera_{camera.camera_index:02}.png"
                print(f"Write to {image_path}")
                save_color_image(color_image, image_path, write_shape=True)

        dt_s = time.time() - now
        dt_ms = dt_s * 1000
        freq = 1 / dt_s
        print(f"Latency (ms): {dt_ms:.2f}\tFrequency: {freq:.2f}")

        if save_images:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


@dataclass
class OpenCVCameraConfig:
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(30, 640, 480)
    OpenCVCameraConfig(60, 640, 480)
    OpenCVCameraConfig(90, 640, 480)
    OpenCVCameraConfig(30, 1280, 720)
    ```
    """

    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color: str = "rgb"


class OpenCVCamera:
    # TODO(rcadene): improve dosctring
    """
    https://docs.opencv.org/4.x/d0/da7/videoio_overview.html
    https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d

    Example of uage:

    ```python
    camera = OpenCVCamera(2)
    color_image = camera.capture_image()
    ```
    """

    def __init__(self, camera_index: int, config: OpenCVCameraConfig | None = None, **kwargs):
        if config is None:
            config = OpenCVCameraConfig()
        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)

        self.camera_index = camera_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color = config.color

        if self.color not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {self.color} is provided.")

        if self.camera_index is None:
            raise ValueError(
                f"`camera_index` is expected to be one of these available cameras {OpenCVCamera.AVAILABLE_CAMERAS_INDICES}, but {camera_index} is provided instead."
            )

        self.camera = None
        self.is_connected = False
        self.threads = {}
        self.results = {}
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise ValueError(f"Camera {self.camera_index} is already connected.")

        # First create a temporary camera trying to access `camera_index`,
        # and verify it is a valid camera by calling `isOpened`.
        tmp_camera = cv2.VideoCapture(self.camera_index)
        is_camera_open = tmp_camera.isOpened()
        # Release camera to make it accessible for `find_camera_indices`
        del tmp_camera

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            if self.camera_index not in find_camera_indices():
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {OpenCVCamera.AVAILABLE_CAMERAS_INDICES}, but {self.camera_index} is provided instead."
                )

            raise OSError(f"Can't access camera {self.camera_index}.")

        # Secondly, create the camera that will be used downstream.
        # Note: For some unknown reason, calling `isOpened` blocks the camera which then
        # needs to be re-created.
        self.camera = cv2.VideoCapture(self.camera_index)

        if self.fps:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if self.fps and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise OSError(
                f"Can't set {self.fps=} for camera {self.camera_index}. Actual value is {actual_fps}."
            )
        if self.width and self.width != actual_width:
            raise OSError(
                f"Can't set {self.width=} for camera {self.camera_index}. Actual value is {actual_width}."
            )
        if self.height and self.height != actual_height:
            raise OSError(
                f"Can't set {self.height=} for camera {self.camera_index}. Actual value is {actual_height}."
            )

        self.is_connected = True

    def read(self, temporary_color: str | None = None) -> np.ndarray:
        if not self.is_connected:
            self.connect()

        start_time = time.perf_counter()

        ret, color_image = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color = self.color if temporary_color is None else temporary_color

        if requested_color not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {requested_color} is provided.")

        # OpenCV uses BGR format as default (blue, green red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        return color_image

    def read_loop(self):
        while True:
            self.results["color_image"] = self.read()

    def async_read(self):
        if "read" not in self.threads:
            self.threads["read"] = Thread(target=self.read_loop, args=())
            self.threads["read"].daemon = True
            self.threads["read"].start()

        num_tries = 0
        while "color_image" not in self.results:
            num_tries += 1
            time.sleep(1/self.fps)
            if num_tries > self.fps:
                if self.threads["read"].ident is None and not self.threads["read"].is_alive():
                    raise Exception("The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.threads[\"read\"].start()` has been called.")

        return self.results["color_image"] #, self.logs["timestamp_utc"]

    def disconnect(self):
        if getattr(self, "camera", None):
            self.camera.release()
            for name in self.threads:
                if self.threads[name].is_alive():
                    # wait for the thread to finish
                    self.threads[name].join()

    def __del__(self):
        self.disconnect()


def save_images_config(config: OpenCVCameraConfig, out_dir: Path):
    cameras = []

    available_cam_ids = find_camera_indices()
    print(f"Available camera indices: {available_cam_ids}")
    for camera_idx in available_cam_ids:
        camera = OpenCVCamera(camera_idx, config)
        cameras.append(camera)

    out_dir = out_dir.parent / f"{out_dir.name}_{config.width}x{config.height}_{config.fps}"
    benchmark_cameras(cameras, out_dir, save_images=True)


def benchmark_config(config: OpenCVCameraConfig, camera_ids: list[int]):
    cameras = [OpenCVCamera(idx, config) for idx in camera_ids]
    benchmark_cameras(cameras)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["save_images", "benchmark"], default="save_images")
    parser.add_argument("--camera-ids", type=int, nargs="*", default=[16, 4, 22, 10])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=str, default=640)
    parser.add_argument("--height", type=str, default=480)
    parser.add_argument("--out-dir", type=Path, default="outputs/benchmark_cameras/opencv/2024_06_22_1727")
    args = parser.parse_args()

    config = OpenCVCameraConfig(args.fps, args.width, args.height)
    # config = OpenCVCameraConfig()
    # config = OpenCVCameraConfig(60, 640, 480)
    # config = OpenCVCameraConfig(90, 640, 480)
    # config = OpenCVCameraConfig(30, 1280, 720)

    if args.mode == "save_images":
        save_images_config(config, args.out_dir)
    elif args.mode == "benchmark":
        benchmark_config(config, args.camera_ids)
    else:
        raise ValueError(args.mode)
