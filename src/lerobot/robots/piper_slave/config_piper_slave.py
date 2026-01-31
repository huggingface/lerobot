#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_slave")
@dataclass
class PIPERSlaveConfig(RobotConfig):
    port: str = "can_left"
    read_only: bool = True
    # # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                fps=30,
                width=480,
                height=640,
                rotation=-90,
            ),
            "ground": OpenCVCameraConfig(
                index_or_path="/dev/video2",
                fps=30,
                width=480,
                height=640,
                rotation=90,
            ),
        }
    )
