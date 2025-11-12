from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig



@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface: str = "can0"
    bitrate: int = 1_000_000
    # Piper SDK returns 6 joints; keep order stable
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i+1}" for i in range(6)])
    # Optional sign flips applied symmetrically to obs/actions (length must match joints)
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])
    # Allow teleop joints (e.g., SO101) to reference Piper joints directly by name
    joint_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "joint_1",
            "shoulder_lift": "joint_2",
            "elbow_flex": "joint_3",
            "wrist_flex": "joint_5",
            "wrist_roll": "joint_6",
        }
    )
    # Expose gripper as "gripper.pos" in mm if True
    include_gripper: bool = False
    # Optional cameras; leave empty when not used
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": OpenCVCameraConfig(
                index_or_path=4, 
                width=640, 
                height=480, 
                fps=30, 
                fourcc="MJPG"
            )
        }
    )
    # When False, expose normalized [-100,100] joint percents; when True, degrees/mm
    use_degrees: bool = True
    # Timeout in seconds to wait for SDK EnablePiper during connect
    enable_timeout: float = 5.0