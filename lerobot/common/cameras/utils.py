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
