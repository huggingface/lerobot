from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("ax_arm")
@dataclass
class AXArmConfig(RobotConfig):
    """Configuration for a 4-DoF Dynamixel AX-series (Protocol 1.0) arm."""

    # Serial port the arm is connected to (e.g. "/dev/tty.usbserial-AL02L1E0").
    port: str

    disable_torque_on_disconnect: bool = True

    # Limits the magnitude of relative positional targets for safety. Scalar (same for all motors) or a
    # dict mapping motor names to their own cap.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Normalize body joints to [-100, 100] (and gripper to [0, 100]) when False, or to degrees when True.
    use_degrees: bool = False
