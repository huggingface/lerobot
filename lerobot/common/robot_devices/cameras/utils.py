from typing import Protocol

import numpy as np

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)


# Defines a camera type
class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> list[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

            cameras[key] = IntelRealSenseCamera(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return cameras


def make_camera(camera_type, **kwargs) -> Camera:
    if camera_type == "opencv":
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

        config = OpenCVCameraConfig(**kwargs)
        return OpenCVCamera(config)

    elif camera_type == "intelrealsense":
        from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

        config = IntelRealSenseCameraConfig(**kwargs)
        return IntelRealSenseCamera(config)

    else:
        raise ValueError(f"The camera type '{camera_type}' is not valid.")
