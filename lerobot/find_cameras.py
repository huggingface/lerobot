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

import argparse
import concurrent.futures
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from lerobot.common.cameras.configs import ColorMode
from lerobot.common.cameras.intel.camera_realsense import RealSenseCamera
from lerobot.common.cameras.intel.configuration_realsense import RealSenseCameraConfig
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> List[Dict[str, Union[str, int, float, List[str], None]]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: List[Dict[str, Union[str, int, float, List[str], None]]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras(raise_when_empty=False)
        for cam_info in opencv_cameras:
            cam_info.setdefault("name", f"OpenCV Camera @ {cam_info['id']}")
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> List[Dict[str, Union[str, int, float, List[str], None]]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: List[Dict[str, Union[str, int, float, List[str], None]]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras(raise_when_empty=False)
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_all_cameras() -> List[Dict[str, Union[str, int, float, List[str], None]]]:
    """
    Finds all available cameras (OpenCV and RealSense) plugged into the system.

    Returns:
        A unified list of all available cameras with their metadata.
    """

    all_opencv_cameras_info = find_all_opencv_cameras()
    all_realsense_cameras_info = find_all_realsense_cameras()

    all_cameras_info = all_opencv_cameras_info + all_realsense_cameras_info

    if not all_cameras_info:
        logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i + 1}:")
            for key, value in cam_info.items():
                print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)

    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: Union[str, int],
    images_dir: Path,
    camera_type: str,
):
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")


def save_images_from_all_cameras(
    output_dir: Union[str, Path],
    width: int = 640,
    height: int = 480,
    record_time_s: int = 2,
):
    """
    Connects to all detected cameras and saves a few images from each.

    Args:
        output_dir: Directory to save images.
        width: Target width.
        height: Target height.
        record_time_s: Duration in seconds to record images.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        logger.info(f"Output directory {output_dir} exists. Removing previous content.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")

    all_camera_metadata = find_all_cameras()
    if not all_camera_metadata:
        logger.warning("No cameras detected. Cannot save images.")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        cam_type = cam_meta.get("type")
        cam_id = cam_meta.get("id")
        instance = None

        try:
            if cam_type == "OpenCV":
                cv_config = OpenCVCameraConfig(
                    index_or_path=cam_id, color_mode=ColorMode.RGB, width=width, height=height, fps=30
                )
                instance = OpenCVCamera(cv_config)
            elif cam_type == "RealSense":
                rs_config = RealSenseCameraConfig(
                    serial_number=str(cam_id), width=width, height=height, fps=30
                )
                instance = RealSenseCamera(rs_config)
            else:
                logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
                continue

            if instance:
                logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
                instance.connect()
                cameras_to_use.append({"instance": instance, "meta": cam_meta})
        except Exception as e:
            logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
            if instance and instance.is_connected:
                instance.disconnect()

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    logger.info(f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras.")
    frame_index = 0
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras_to_use) * 2) as executor:
        try:
            while time.perf_counter() - start_time < record_time_s:
                futures = []

                for cam_dict in cameras_to_use:
                    cam = cam_dict["instance"]
                    meta = cam_dict["meta"]
                    cam_type_str = str(meta.get("type", "unknown"))
                    cam_id_str = str(meta.get("id", "unknown"))

                    try:
                        image_data = cam.read()

                        if image_data is None:
                            logger.warning(
                                f"No frame received from {cam_type_str} camera {cam_id_str} for frame {frame_index}."
                            )
                            continue

                        futures.append(
                            executor.submit(
                                save_image,
                                image_data,
                                cam_id_str,
                                output_dir,
                                cam_type_str,
                            )
                        )

                    except TimeoutError:
                        logger.warning(
                            f"Timeout reading from {cam_type_str} camera {cam_id_str} for frame {frame_index}."
                        )
                    except Exception as e:
                        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")

                concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")
        finally:
            print("\nFinalizing image saving...")
            executor.shutdown(wait=True)
            logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
            for cam_dict in cameras_to_use:
                try:
                    if cam_dict["instance"] and cam_dict["instance"].is_connected:
                        cam_dict["instance"].disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")
            logger.info(f"Image capture finished. Images saved to {output_dir}")


# NOTE(Steven): Add CLI for finding-cameras of just one type
# NOTE(Steven): Check why opencv detects realsense cameras
# NOTE(Steven): Check why saving cameras is buggy
# NOTE(Steven): Check how to deal with different resolutions macos
# NOTE(Steven): Ditch width height resolutions in favor of defaults
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # List cameras command
    list_parser = subparsers.add_parser(
        "list-cameras", help="Shows all connected cameras (OpenCV and RealSense)"
    )
    list_parser.set_defaults(func=lambda args: find_all_cameras())

    # Capture images command
    capture_parser = subparsers.add_parser("capture-images", help="Saves images from all detected cameras")
    capture_parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    capture_parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Set the capture width for all cameras. If not provided, uses camera defaults.",
    )
    capture_parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Set the capture height for all cameras. If not provided, uses camera defaults.",
    )
    capture_parser.add_argument(
        "--record-time-s",
        type=float,
        default=10.0,
        help="Set the number of seconds to record frames. Default: 2.0 seconds.",
    )
    capture_parser.set_defaults(
        func=lambda args: save_images_from_all_cameras(
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
            record_time_s=args.record_time_s,
        )
    )

    args = parser.parse_args()
    args.func(args)
