from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation


@RobotConfig.register_subclass("meca")
@dataclass
@RobotConfig.register_subclass("meca")
@dataclass
class MecaConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=0,
                width=640,
                height=360,
                fps=30,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION,
            ),
            "cam_2": OpenCVCameraConfig(
                index_or_path=3,
                width=640,
                height=360,
                fps=260,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.ROTATE_180,
            ),
        }
    )
    ip: str = "192.168.0.100"   # default Meca500 IP
    port: int = 3000           # default control port

