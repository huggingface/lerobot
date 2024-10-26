from dataclasses import dataclass, field, replace

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError


@dataclass
class Ned2RobotConfig:
    """
    Example of usage:
    ```python
    Ned2RobotConfig()
    ```
    """

    # Define all components of the robot
    robot_type: str = "ned2"
    leader_arms: dict[str, list] = field(default_factory=lambda: {})
    follower_arms: dict[str, list] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.robot_type not in ["ned2"]:
            raise ValueError(f"Provided robot type ({self.robot_type}) is not supported.")


class Ned2Robot:
    """Niryo Ned 2 robot interface"""

    def __init__(self, config: Ned2RobotConfig | None = None, **kwargs):
        if config is None:
            config = Ned2RobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = None
        self.logs = {}

        # Needed for dataset v2
        action_names = [f"{arm}_{motor}" for arm, bus in self.leader_arms.items() for motor in bus.motors]
        state_names = [f"{arm}_{motor}" for arm, bus in self.follower_arms.items() for motor in bus.motors]
        self.names = {
            "action": action_names,
            "observation.state": state_names,
        }

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self) -> None:
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Ned2Robot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "Ned2Robot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        pass  # TODO

    def run_calibration(self):
        pass  # TODO

    def teleop_step(self, record_data=False):
        pass  # TODO

    def capture_observation(self):
        pass  # TODO

    def send_action(self, action):
        pass  # TODO

    def disconnect(self):
        pass  # TODO
