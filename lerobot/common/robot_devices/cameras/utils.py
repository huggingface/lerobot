from typing import Protocol

import numpy as np

from lerobot.common.robot_devices.cameras.configs import CameraConfig
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


# Defines a camera type
class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...


def build_cameras(camera_configs: dict[str, CameraConfig]):
    cameras = {}
    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            cameras[key] = OpenCVCamera(cfg)
        elif cfg.type == "intelrealsense":
            cameras[key] = IntelRealSenseCamera(cfg)
        else:
            raise ValueError(f"{cfg.type} type is not found.")
    return cameras
