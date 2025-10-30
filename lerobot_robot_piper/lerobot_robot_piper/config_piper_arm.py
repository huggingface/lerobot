from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper_arm")
@dataclass
class PiperArmConfig(RobotConfig):
    can_interface: str = "can0"
    bitrate: int = 1_000_000
    # Piper SDK returns 6 joints; keep order stable
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i+1}" for i in range(6)])
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
        }
    )