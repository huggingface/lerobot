#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import platform
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation

IndexOrPath: TypeAlias = int | Path


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .intel.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int:
    import cv2

    return {
        Cv2Rotation.ROTATE_270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        Cv2Rotation.ROTATE_90: cv2.ROTATE_90_CLOCKWISE,
        Cv2Rotation.ROTATE_180: cv2.ROTATE_180,
    }.get(rotation)


def get_cv2_backend() -> int:
    import cv2

    return {
        "Linux": cv2.CAP_DSHOW,
        "Windows": cv2.CAP_AVFOUNDATION,
        "Darwin": cv2.CAP_ANY,
    }.get(platform.system(), cv2.CAP_V4L2)


def save_image(img_array: np.ndarray, camera_index: int, frame_index: int, images_dir: Path):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


# NOTE(Steven): This should be use with both cameras implementations
# def save_images_from_cameras(
#     images_dir: Path,
#     camera_idx_or_paths: list[IndexOrPath] | None = None,
#     fps: int | None = None,
#     width: int | None = None,
#     height: int | None = None,
#     record_time_s: int = 2,
# ):
#     """
#     Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
#     associated to a given camera index.
#     """
#     if not camera_idx_or_paths:
#         camera_idx_or_paths = OpenCVCamera.find_cameras()
#         if len(camera_idx_or_paths) == 0:
#             raise RuntimeError(
#                 "Not a single camera was detected. Try re-plugging, or re-installing `opencv-python`, "
#                 "or your camera driver, or make sure your camera is compatible with opencv."
#             )

#     print("Connecting cameras")
#     cameras = []
#     for idx_or_path in camera_idx_or_paths:
#         config = OpenCVCameraConfig(index_or_path=idx_or_path, fps=fps, width=width, height=height)
#         camera = OpenCVCamera(config)
#         camera.connect()
#         print(
#             f"OpenCVCamera({camera.index_or_path}, fps={camera.fps}, width={camera.capture_width}, "
#             f"height={camera.capture_height}, color_mode={camera.color_mode})"
#         )
#         cameras.append(camera)

#     images_dir = Path(images_dir)
#     if images_dir.exists():
#         shutil.rmtree(
#             images_dir,
#         )
#     images_dir.mkdir(parents=True, exist_ok=True)

#     print(f"Saving images to {images_dir}")
#     frame_index = 0
#     start_time = time.perf_counter()
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         while True:
#             now = time.perf_counter()

#             for camera in cameras:
#                 # If we use async_read when fps is None, the loop will go full speed, and we will endup
#                 # saving the same images from the cameras multiple times until the RAM/disk is full.
#                 image = camera.read() if fps is None else camera.async_read()

#                 executor.submit(
#                     save_image,
#                     image,
#                     camera.camera_index,
#                     frame_index,
#                     images_dir,
#                 )

#             if fps is not None:
#                 dt_s = time.perf_counter() - now
#                 busy_wait(1 / fps - dt_s)

#             print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

#             if time.perf_counter() - start_time > record_time_s:
#                 break

#             frame_index += 1

#     print(f"Images have been saved to {images_dir}")
#     # NOTE(Steven): Cameras don't get disconnected


# # NOTE(Steven): Update this to be valid for both cameras type
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
#     )
#     parser.add_argument(
#         "--camera-ids",
#         type=int,
#         nargs="*",
#         default=None,
#         help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
#     )
#     parser.add_argument(
#         "--fps",
#         type=int,
#         default=None,
#         help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
#     )
#     parser.add_argument(
#         "--width",
#         type=str,
#         default=None,
#         help="Set the width for all cameras. If not provided, use the default width of each camera.",
#     )
#     parser.add_argument(
#         "--height",
#         type=str,
#         default=None,
#         help="Set the height for all cameras. If not provided, use the default height of each camera.",
#     )
#     parser.add_argument(
#         "--images-dir",
#         type=Path,
#         default="outputs/images_from_opencv_cameras",
#         help="Set directory to save a few frames for each camera.",
#     )
#     parser.add_argument(
#         "--record-time-s",
#         type=float,
#         default=4.0,
#         help="Set the number of seconds used to record the frames. By default, 2 seconds.",
#     )
#     args = parser.parse_args()
#     save_images_from_cameras(**vars(args))


### Realsense

# def find_realsense_cameras(raise_when_empty: bool = True) -> list[dict]:
#     """
#     Find the names and the serial numbers of the Intel RealSense cameras
#     connected to the computer.
#     """
#     import pyrealsense2 as rs

#     cameras = []
#     for device in rs.context().query_devices():
#         serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
#         name = device.get_info(rs.camera_info.name)
#         cameras.append(
#             {
#                 "serial_number": serial_number,
#                 "name": name,
#             }
#         )

#     if raise_when_empty and len(cameras) == 0:
#         raise OSError(
#             "Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware."
#         )

#     return cameras


# def save_image(img_array, serial_number, frame_index, images_dir):
#     try:
#         img = Image.fromarray(img_array)
#         path = images_dir / f"camera_{serial_number}_frame_{frame_index:06d}.png"
#         path.parent.mkdir(parents=True, exist_ok=True)
#         img.save(str(path), quality=100)
#         logging.info(f"Saved image: {path}")
#     except Exception as e:
#         logging.error(f"Failed to save image for camera {serial_number} frame {frame_index}: {e}")


# def save_images_from_cameras(
#     images_dir: Path,
#     serial_numbers: list[int] | None = None,
#     fps=None,
#     width=None,
#     height=None,
#     record_time_s=2,
# ):
#     """
#     Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
#     associated to a given serial number.
#     """
#     if serial_numbers is None or len(serial_numbers) == 0:
#         camera_infos = find_realsense_cameras()
#         serial_numbers = [cam["serial_number"] for cam in camera_infos]

#     import cv2

#     print("Connecting cameras")
#     cameras = []
#     for cam_sn in serial_numbers:
#         print(f"{cam_sn=}")
#         config = RealSenseCameraConfig(serial_number=cam_sn, fps=fps, width=width, height=height)
#         camera = RealSenseCamera(config)
#         camera.connect()
#         print(
#             f"RealSenseCamera({camera.serial_number}, fps={camera.fps}, width={camera.capture_width}, height={camera.capture_height}, color_mode={camera.color_mode})"
#         )
#         cameras.append(camera)

#     images_dir = Path(images_dir)
#     if images_dir.exists():
#         shutil.rmtree(
#             images_dir,
#         )
#     images_dir.mkdir(parents=True, exist_ok=True)

#     print(f"Saving images to {images_dir}")
#     frame_index = 0
#     start_time = time.perf_counter()
#     try:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             while True:
#                 now = time.perf_counter()

#                 for camera in cameras:
#                     # If we use async_read when fps is None, the loop will go full speed, and we will end up
#                     # saving the same images from the cameras multiple times until the RAM/disk is full.
#                     image = camera.read() if fps is None else camera.async_read()
#                     if image is None:
#                         print("No Frame")

#                     bgr_converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                     executor.submit(
#                         save_image,
#                         bgr_converted_image,
#                         camera.serial_number,
#                         frame_index,
#                         images_dir,
#                     )

#                 if fps is not None:
#                     dt_s = time.perf_counter() - now
#                     busy_wait(1 / fps - dt_s)

#                 if time.perf_counter() - start_time > record_time_s:
#                     break

#                 print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

#                 frame_index += 1
#     finally:
#         print(f"Images have been saved to {images_dir}")
#         for camera in cameras:
#             camera.disconnect()


# def find_serial_number_from_name(name):
#     camera_infos = find_realsense_cameras()
#     camera_names = [cam["name"] for cam in camera_infos]
#     this_name_count = Counter(camera_names)[name]
#     if this_name_count > 1:
#         # TODO(aliberts): Test this with multiple identical cameras (Aloha)
#         raise ValueError(
#             f"Multiple {name} cameras have been detected. Please use their serial number to instantiate them."
#         )

#     name_to_serial_dict = {cam["name"]: cam["serial_number"] for cam in camera_infos}
#     cam_sn = name_to_serial_dict[name]

#     return cam_sn

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Save a few frames using `RealSenseCamera` for all cameras connected to the computer, or a selected subset."
#     )
#     parser.add_argument(
#         "--serial-numbers",
#         type=int,
#         nargs="*",
#         default=None,
#         help="List of serial numbers used to instantiate the `RealSenseCamera`. If not provided, find and use all available camera indices.",
#     )
#     parser.add_argument(
#         "--fps",
#         type=int,
#         default=30,
#         help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
#     )
#     parser.add_argument(
#         "--width",
#         type=str,
#         default=640,
#         help="Set the width for all cameras. If not provided, use the default width of each camera.",
#     )
#     parser.add_argument(
#         "--height",
#         type=str,
#         default=480,
#         help="Set the height for all cameras. If not provided, use the default height of each camera.",
#     )
#     parser.add_argument(
#         "--images-dir",
#         type=Path,
#         default="outputs/images_from_intelrealsense_cameras",
#         help="Set directory to save a few frames for each camera.",
#     )
#     parser.add_argument(
#         "--record-time-s",
#         type=float,
#         default=2.0,
#         help="Set the number of seconds used to record the frames. By default, 2 seconds.",
#     )
#     args = parser.parse_args()
#     save_images_from_cameras(**vars(args))
