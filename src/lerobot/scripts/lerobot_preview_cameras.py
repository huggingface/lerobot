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
Live preview of camera feeds using OpenCV windows.

Example:

```shell
# Preview all detected OpenCV cameras
lerobot-preview-cameras

# Preview specific camera indices
lerobot-preview-cameras 0 1 2
```
"""

import argparse
import sys

import cv2

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def preview_cameras(indices_or_paths: list[int | str]) -> None:
    """Connect to each camera index/path and show live feed in separate windows."""
    cameras: list[OpenCVCamera] = []
    window_names: list[str] = []

    try:
        for idx in indices_or_paths:
            config = OpenCVCameraConfig(
                index_or_path=idx,
                color_mode=ColorMode.RGB,
            )
            cam = OpenCVCamera(config)
            cam.connect(warmup=True)
            cameras.append(cam)
            name = f"Camera {idx}"
            window_names.append(name)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        print("Showing camera preview. Press 'q' in any window to quit.")
        while True:
            for i, (cam, name) in enumerate(zip(cameras, window_names)):
                frame = cam.read()
                if frame is None:
                    continue
                # OpenCV imshow expects BGR
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(name, bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for name in window_names:
            cv2.destroyWindow(name)
        for cam in cameras:
            if cam.is_connected:
                cam.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live preview of OpenCV camera feeds in separate windows.",
    )
    parser.add_argument(
        "indices",
        type=int,
        nargs="*",
        help="Camera indices to preview (e.g. 0 1 2). If omitted, discovers and previews all OpenCV cameras.",
    )
    args = parser.parse_args()

    if args.indices:
        indices = args.indices
    else:
        # Discover all OpenCV cameras
        try:
            found = OpenCVCamera.find_cameras()
        except Exception as e:
            print(f"Error discovering cameras: {e}", file=sys.stderr)
            sys.exit(1)
        if not found:
            print("No OpenCV cameras detected. Try: lerobot-find-cameras opencv", file=sys.stderr)
            sys.exit(1)
        indices = [c["id"] for c in found]
        print(f"Previewing cameras: {indices}")

    preview_cameras(indices)


if __name__ == "__main__":
    main()
