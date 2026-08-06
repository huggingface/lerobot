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

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    """Instantiate one [`~cameras.Camera`] per entry in a mapping of configs.

    Dispatches on each config's registered `type` to build the matching backend class. This only
    constructs the camera objects; call [`~cameras.Camera.connect`] on each before use.

    Args:
        camera_configs (`dict[str, CameraConfig]`):
            Camera configs keyed by the name each camera should be identified by, e.g. in a robot's
            observation features.

    Returns:
        `dict[str, Camera]`: A camera instance per key, in the same order as `camera_configs`.

    Raises:
        ValueError: If a config's type is not a known backend and building it via the generic device
            factory also fails.

    Example:
        ```python
        >>> from lerobot.cameras.opencv import OpenCVCameraConfig
        >>> from lerobot.cameras.utils import make_cameras_from_configs
        >>> configs = {"top": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)}
        >>> cameras = make_cameras_from_configs(configs)
        >>> list(cameras.keys())
        ['top']
        ```
    """
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        # TODO(Steven): Consider just using the make_device_from_device_class for all types
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        elif cfg.type == "zmq":
            from .zmq.camera_zmq import ZMQCamera

            cameras[key] = ZMQCamera(cfg)

        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating camera {key} with config {cfg}: {e}") from e

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    """Map a [`~cameras.Cv2Rotation`] to the OpenCV rotation flag `cv2.rotate` expects.

    Args:
        rotation (`Cv2Rotation`):
            The configured rotation.

    Returns:
        `int | None`: The matching `cv2.ROTATE_*` constant, or `None` for [`~cameras.Cv2Rotation.NO_ROTATION`].
    """
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None
