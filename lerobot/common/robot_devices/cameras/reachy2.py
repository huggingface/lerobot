"""
Wrapper for Reachy2 camera from sdk
"""

from dataclasses import dataclass

import numpy as np
from reachy2_sdk.media.camera import CameraView
from reachy2_sdk.media.camera_manager import CameraManager


@dataclass
class ReachyCameraConfig:
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    rotation: int | None = None
    mock: bool = False


class ReachyCamera:
    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        image_type: str,
        config: ReachyCameraConfig | None = None,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.image_type = image_type
        self.name = name
        self.config = config
        self.cam_manager = None
        self.is_connected = False
        self.logs = {}

    def connect(self):
        if not self.is_connected:
            self.cam_manager = CameraManager(host=self.host, port=self.port)
            self.cam_manager.initialize_cameras()  # FIXME: maybe we should not re-initialize
            self.is_connected = True

    def read(self) -> np.ndarray:
        if not self.is_connected:
            self.connect()

        if self.name == "teleop" and hasattr(self.cam_manager, "teleop"):
            if self.image_type == "left":
                return self.cam_manager.teleop.get_frame(CameraView.LEFT)
                # return self.cam_manager.teleop.get_compressed_frame(CameraView.LEFT)
            elif self.image_type == "right":
                return self.cam_manager.teleop.get_frame(CameraView.RIGHT)
                # return self.cam_manager.teleop.get_compressed_frame(CameraView.RIGHT)
            else:
                return None
        elif self.name == "depth" and hasattr(self.cam_manager, "depth"):
            if self.image_type == "depth":
                return self.cam_manager.depth.get_depth_frame()
            elif self.image_type == "rgb":
                return self.cam_manager.depth.get_frame()
                # return self.cam_manager.depth.get_compressed_frame()
            else:
                return None
        return None
