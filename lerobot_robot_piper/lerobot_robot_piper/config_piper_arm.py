from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig

@RobotConfig.register_subclass("piper_arm")
@dataclass
class PiperArmConfig(RobotConfig):
    can_interface: str = "can0"
    bitrate: int = 1000000
    joint_names: list[str] = field(default_factory=lambda: ["j1", "j2", "j3", "j4", "j5", "j6"])
    joint_offsets: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_gains: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    joint_velocities: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    joint_torques: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    joint_positions: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_efforts: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_velocities: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    joint_torques: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    joint_positions: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_efforts: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])