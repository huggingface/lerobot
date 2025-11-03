from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface: str = "can0"
    bitrate: int = 1_000_000
    # Timeout in seconds to wait for SDK EnablePiper during connect
    enable_timeout: float = 10.0
    # Piper SDK returns 6 joints; keep order stable
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i+1}" for i in range(6)])
    # Optional sign flips applied symmetrically to obs/actions (length must match joints)
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])
    # Expose gripper as "gripper.pos" in mm if True
    include_gripper: bool = False
    # Optional cameras; leave empty when not used
    cameras: dict[str, CameraConfig] = field(default_factory=dict)