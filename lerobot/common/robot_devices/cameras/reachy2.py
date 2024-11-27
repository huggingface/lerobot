"""
Wrapper for Reachy2 camera from sdk
"""

from dataclasses import dataclass, replace

import cv2
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

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3


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
        if config is None:
            config = ReachyCameraConfig()

        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)

        self.host = host
        self.port = port
        self.width = config.width
        self.height = config.height
        self.channels = config.channels
        self.fps = config.fps
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

        frame = None

        if self.name == "teleop" and hasattr(self.cam_manager, "teleop"):
            if self.image_type == "left":
                frame = self.cam_manager.teleop.get_frame(CameraView.LEFT)
            elif self.image_type == "right":
                frame = self.cam_manager.teleop.get_frame(CameraView.RIGHT)
        elif self.name == "depth" and hasattr(self.cam_manager, "depth"):
            if self.image_type == "depth":
                frame = self.cam_manager.depth.get_depth_frame()
            elif self.image_type == "rgb":
                frame = self.cam_manager.depth.get_frame()

        if frame is None:
            return None

        if frame is not None and self.config.color_mode == "rgb":
            img, timestamp = frame
            frame = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), timestamp)

        return frame
