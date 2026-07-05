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

Examples:

```shell
lerobot-find-cameras
lerobot-find-cameras opencv --live   # live tiled preview; 'q'/ESC quit, 's' snapshot
```

Note: --live requires an OpenCV build with GUI support (opencv-python, not
opencv-python-headless).
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot-find-cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
import concurrent.futures
import logging
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from lerobot.cameras import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

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


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv").
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
    img_array: np.ndarray,
    camera_identifier: str | int,
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
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=True)
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


def _check_cv2_gui_available() -> None:
    """Verify the installed OpenCV build has GUI (highgui) support.

    The declared dependency is ``opencv-python-headless``, which lacks
    ``imshow``/``namedWindow``/``waitKey``. Calling them raises ``cv2.error``.
    This probes that capability up front so ``--live`` fails with an actionable
    message instead of a cryptic traceback mid-loop.

    Raises:
        RuntimeError: If the OpenCV build has no GUI support.
    """
    probe = "__lerobot_gui_probe__"
    try:
        cv2.namedWindow(probe, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(probe)
        cv2.waitKey(1)
    except cv2.error as e:
        raise RuntimeError(
            "Live preview (--live) requires OpenCV GUI support, but the installed "
            "build is headless (opencv-python-headless). Install the GUI build:\n"
            "    pip uninstall -y opencv-python-headless\n"
            "    pip install opencv-python\n"
            f"(original error: {e})"
        ) from e


def build_camera_grid(
    frames: list[np.ndarray | None],
    labels: list[str],
    tile_size: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Assemble per-camera BGR frames into a single tiled grid image.

    Computes a near-square grid (``cols = ceil(sqrt(n))``, ``rows = ceil(n/cols)``),
    resizes every frame to a uniform tile, overlays its label, and pads unused
    cells with black tiles so the grid is a perfect rectangle.

    Args:
        frames: One entry per camera. ``None`` (failed read) becomes a black tile.
        labels: Per-camera label text, same length and order as ``frames``.
        tile_size: Target ``(height, width)`` of each tile.

    Returns:
        A single BGR ``uint8`` image of shape ``(rows * tile_h, cols * tile_w, 3)``.
    """
    tile_h, tile_w = tile_size
    n = len(frames)
    if n == 0:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    tiles: list[np.ndarray] = []
    for i in range(rows * cols):
        if i < n and frames[i] is not None:
            tile = cv2.resize(frames[i], (tile_w, tile_h))
            if tile.ndim == 2:  # grayscale safety
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            tile = np.ascontiguousarray(tile)
            label = labels[i]
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            label = f"{labels[i]} (no signal)" if i < n else ""

        if label:
            cv2.putText(
                tile,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        tiles.append(tile)

    row_imgs = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.vstack(row_imgs)


def _save_snapshot(frames: list[np.ndarray | None], labels: list[str], output_dir: Path) -> None:
    """Save current live frames (BGR) as PNG files into ``output_dir``.

    Frames are converted BGR->RGB before delegating to :func:`save_image`, which
    expects RGB. Filenames match the timed-capture path (``<type>_<id>.png``) and
    overwrite on repeated snapshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, label in zip(frames, labels, strict=False):
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_type, _, cam_id = label.partition(" ")
        save_image(rgb, cam_id, output_dir, cam_type)
    logger.info(f"Snapshot saved to {output_dir}")


def show_live_preview(
    output_dir: Path,
    camera_type: str | None = None,
    tile_size: tuple[int, int] = (480, 640),
):
    """Open all matching cameras and show a live tiled preview in one window.

    Keys: ``q``/``ESC`` quit; ``s`` saves a snapshot of all current frames to
    ``output_dir``. Used instead of the timed capture when ``--live`` is passed.

    Args:
        output_dir: Directory snapshots are written to (created on demand).
        camera_type: Optional filter ("opencv" / "realsense"); None uses all.
        tile_size: Per-tile ``(height, width)`` for the grid.
    """
    _check_cv2_gui_available()

    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)
    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot start live preview.")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting live preview.")
        return

    labels = [f"{c['meta'].get('type')} {c['meta'].get('id')}" for c in cameras_to_use]
    last_good: list[np.ndarray | None] = [None] * len(cameras_to_use)

    window = "lerobot-find-cameras (live)  |  q/ESC: quit   s: snapshot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    logger.info("Live preview started. Press 'q' or ESC to quit, 's' to snapshot.")

    try:
        while True:
            frames: list[np.ndarray | None] = []
            for i, cam_dict in enumerate(cameras_to_use):
                try:
                    # Request BGR directly so frames are imshow-ready (no conversion).
                    frame = cam_dict["instance"].read(color_mode=ColorMode.BGR)
                    last_good[i] = frame
                    frames.append(frame)
                except Exception as e:
                    logger.debug(f"Read failed for {labels[i]}: {e}")
                    frames.append(last_good[i])  # last good frame, else None -> black tile

            grid = build_camera_grid(frames, labels, tile_size=tile_size)
            cv2.imshow(window, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("s"):
                _save_snapshot(frames, labels, output_dir)
    except KeyboardInterrupt:
        logger.info("Live preview interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # flush window events on macOS/Qt
        cleanup_cameras(cameras_to_use)
        logger.info("Live preview finished.")


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
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
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
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show a live tiled preview of all matching cameras instead of capturing for "
        "--record-time-s. Press 'q'/ESC to quit, 's' to snapshot. Requires an OpenCV GUI "
        "build (opencv-python, not opencv-python-headless).",
    )
    args = parser.parse_args()

    if args.live:
        try:
            show_live_preview(output_dir=args.output_dir, camera_type=args.camera_type)
        except RuntimeError as e:
            logger.error(str(e))
            raise SystemExit(1) from e
    else:
        save_images_from_all_cameras(
            output_dir=args.output_dir,
            record_time_s=args.record_time_s,
            camera_type=args.camera_type,
        )


if __name__ == "__main__":
    main()
