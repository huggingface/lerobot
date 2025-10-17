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

"""
Helper to find the camera devices available in your system.

Example:

```shell
lerobot-find-cameras
```
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot-find-cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cameras.utils import get_image_modality_key
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.zed.camera_zed import ZedCamera

from lerobot.cameras.zed.camera_zed import ZedCameraConfig

logger = logging.getLogger(__name__)


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info

def find_all_zed_cameras() -> list[dict[str, Any]]:
    """
    Finds all available ZED cameras plugged into the system.

    Returns:
        A list of all available ZED cameras with their metadata.
    """
    all_zed_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for ZED cameras...")
    try:
        zed_cameras = ZedCamera.find_cameras()
        for cam_info in zed_cameras:
            all_zed_cameras_info.append(cam_info)
        logger.info(f"Found {len(zed_cameras)} ZED cameras.")
    except ImportError:
        logger.warning("Skipping ZED camera search: pyzed library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding ZED cameras: {e}")

    return all_zed_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense", "zed" or "opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())
    if camera_type_filter is None or camera_type_filter == "zed":
        all_cameras_info.extend(find_all_zed_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    image_data: np.ndarray | dict[str, np.ndarray],
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
    modality: str | None = None,
):
    """
    Saves image data to disk using Pillow. Supports multiple modalities.

    Args:
        image_data: Single image array or dictionary of modality-keyed images
        camera_identifier: Unique identifier for the camera
        images_dir: Directory where images will be saved
        camera_type: Type of camera (e.g., 'zed', 'opencv')
        modality: Explicit modality type. If None and image_data is a dict,
                 saves all modalities with automatic key detection.

    Note:
        Supported modalities and their handling:
        - 'gray': Grayscale images, saved as 8-bit PNG
        - 'rgb': RGB images, saved as standard PNG
        - 'rgba': RGBA images, saved as PNG with alpha
        - 'depth': Depth maps, saved as 16-bit PNG
        - 'ir': Infrared images, saved as 8-bit PNG
    """
    try:
        # Handle dictionary input (multiple modalities)
        if isinstance(image_data, dict):
            futures = []
            for mod, img_array in image_data.items():
                # Recursively call save_image for each modality
                future = save_image(
                    img_array, camera_identifier, images_dir, camera_type, mod
                )
                if future:
                    futures.append(future)
            return futures

        # Handle single image array
        img_array = image_data
        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")

        # Auto-detect modality if not explicitly provided
        if modality is None:
            modality = get_image_modality_key(img_array)

        # Process based on modality
        if modality == "depth":
            # Depth image processing
            if img_array.dtype != np.uint16:
                img_array = img_array.astype(np.uint16)
            img = Image.fromarray(img_array, mode="I;16")
            filename_prefix = f"{camera_type.lower()}_{safe_identifier}_depth"

        elif modality == "gray":
            # Grayscale image processing
            if img_array.dtype != np.uint8:
                # Normalize to 0-255 range for 8-bit grayscale
                if img_array.dtype in [np.float32, np.float64]:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode="L")
            filename_prefix = f"{camera_type.lower()}_{safe_identifier}_gray"

        elif modality == "rgb":
            # RGB image processing
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode="RGB")
            filename_prefix = f"{camera_type.lower()}_{safe_identifier}_rgb"

        elif modality == "rgba":
            # RGBA image processing
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode="RGBA")
            filename_prefix = f"{camera_type.lower()}_{safe_identifier}_rgba"

        elif modality == "ir":
            # Infrared image processing (similar to grayscale)
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode="L")
            filename_prefix = f"{camera_type.lower()}_{safe_identifier}_ir"

        else:
            # Fallback for unknown modalities
            logger.warning(f"Unknown modality '{modality}', saving as RGB")
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode="RGB")
            filename_prefix = f"{camera_type.lower()}_{safe_identifier}_{modality}"

        filename = f"{filename_prefix}.png"
        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved {modality} image: {path}")

    except Exception as e:
        logger.error(
            f"Failed to save image for camera {camera_identifier} "
            f"(type {camera_type}, modality={modality}): {e}"
        )


def create_camera_instance(cam_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = RealSenseCamera(rs_config)
        elif cam_type == "ZED":
            zed_config = ZedCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = ZedCamera(zed_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=False)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(
    cam_dict: dict[str, Any], output_dir: Path, current_time: float
) -> concurrent.futures.Future | None:
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))
    logger.info(f"{cam=}\n{meta=}")
    try:
        image_data = cam.read()
        return save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )

    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
        raise e
    return None

def cleanup_cameras(cameras_to_use: list[dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
):
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("realsense" or "opencv").
                            If None, uses all detected cameras.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)
    print(f"{all_camera_metadata=}")
    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return
    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    logger.info(f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras.")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras_to_use) * 2) as executor:
        try:
            while time.perf_counter() - start_time < record_time_s:
                futures = []
                current_capture_time = time.perf_counter()

                for cam_dict in cameras_to_use:
                    future = process_camera_image(cam_dict, output_dir, current_capture_time)
                    if future:
                        futures.append(future)

                if futures:
                    concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")
        finally:
            print("\nFinalizing image saving...")
            executor.shutdown(wait=True)
            cleanup_cameras(cameras_to_use)
            print(f"Image capture finished. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv", "zed"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv', 'zed'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=6.0,
        help="Time duration to attempt capturing frames. Default: 6 seconds.",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
