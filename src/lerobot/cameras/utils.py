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

import os
import platform
from pathlib import Path
from typing import TypeAlias

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
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2

    if rotation == Cv2Rotation.ROTATE_90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rotation == Cv2Rotation.ROTATE_180:
        return cv2.ROTATE_180
    elif rotation == Cv2Rotation.ROTATE_270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return None


def get_cv2_backend() -> int:
    """
    Determine which OpenCV capture backend to use.

    Order of precedence:
      1) LEROBOT_OPENCV_BACKEND env var (one of: any, dshow, msmf, vfw,
         avfoundation, v4l2, gstreamer)
      2) OS default:
         - Windows  -> dshow
         - Darwin   -> avfoundation
         - Linux    -> v4l2
      3) Fallback -> any
    """
    import cv2

    # Mapping from short names to cv2 constants
    _BACKEND_MAP: Dict[str, int] = {
        "any":           cv2.CAP_ANY,
        "dshow":         cv2.CAP_DSHOW,
        "msmf":          cv2.CAP_MSMF,
        "vfw":           cv2.CAP_VFW,
        "avfoundation":  cv2.CAP_AVFOUNDATION,
        "v4l2":          cv2.CAP_V4L2,
        "gstreamer":     cv2.CAP_GSTREAMER,
    }

    env: str = os.getenv("LEROBOT_OPENCV_BACKEND", "").strip().lower()
    if env:
        try:
            return _BACKEND_MAP[env]
        except KeyError:
            valid_opts = ", ".join(_BACKEND_MAP.keys())
            raise ValueError(
                f"Unknown backend '{env}' in LEROBOT_OPENCV_BACKEND; "
                f"valid options are: {valid_opts}"
            )

    system: str = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2

    return cv2.CAP_ANY