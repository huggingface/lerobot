from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
import logging


class KochRobotInitializer:
    """
    Encapsulates the setup of leader/follower arms and cameras
    for the Koch manipulator robot.

    Sub-methods can be tested independently:
      - setup_motors
      - setup_cameras
      - build_robot

    Logging can be enabled/disabled via `enable_logging`.
    """

    def __init__(
        self,
        leader_port: str,
        follower_port: str,
        leader_motors: dict,
        follower_motors: dict,
        camera_params: dict,
        calibration_dir: str = ".cache/calibration/koch",
        enable_logging: bool = False,
    ):
        """
        Args:
          leader_port:    serial port for the leader arm (e.g. "/dev/ttyACM1")
          follower_port:  serial port for the follower arm
          leader_motors:  dict[name->(index, model)] for leader
          follower_motors: dict[name->(index, model)] for follower
          camera_params:  dict[name->(device_id, fps, width, height)]
          calibration_dir: where to store calibration data
          enable_logging: if True, prints debug messages
        """
        self.leader_port = leader_port
        self.follower_port = follower_port
        self.leader_motors = leader_motors
        self.follower_motors = follower_motors
        self.camera_params = camera_params
        self.calibration_dir = calibration_dir

        # configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        fmt = "[%(name)s] %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if enable_logging else logging.WARNING)

        self.logger.debug("Initializer created with:\n"
                          f"  leader_port={leader_port}, follower_port={follower_port}\n"
                          f"  leader_motors={leader_motors.keys()}, follower_motors={follower_motors.keys()}\n"
                          f"  cameras={camera_params.keys()}\n"
                          f"  calibration_dir={calibration_dir}")

        # placeholders
        self.leader_arm = None
        self.follower_arm = None
        self.robot = None

    def setup_motors(self):
        """
        Create motor buses for leader & follower.
        Returns: (leader_arm, follower_arm)
        """
        self.logger.debug("Setting up motor buses...")
        leader_cfg = DynamixelMotorsBusConfig(
            port=self.leader_port,
            motors=self.leader_motors,
        )
        follower_cfg = DynamixelMotorsBusConfig(
            port=self.follower_port,
            motors=self.follower_motors,
        )

        self.leader_arm = DynamixelMotorsBus(leader_cfg)
        self.follower_arm = DynamixelMotorsBus(follower_cfg)

        self.logger.debug("Motor buses initialized.")
        return self.leader_arm, self.follower_arm

    def setup_cameras(self):
        """
        Instantiate OpenCVCameraConfig for each camera.
        Returns: dict[name->OpenCVCameraConfig]
        """
        self.logger.debug("Setting up cameras...")
        cams = {}
        for name, (device, fps, w, h) in self.camera_params.items():
            cams[name] = OpenCVCameraConfig(device, fps=fps, width=w, height=h)
            self.logger.debug(f"  Camera '{name}': device={device}, fps={fps}, {w}×{h}")
        return cams

    def build_robot(self, leader_arm, follower_arm, cameras):
        """
        Build the ManipulatorRobot instance.
        """
        self.logger.debug("Building ManipulatorRobot config...")
        cfg = KochRobotConfig(
            leader_arms={"main": leader_arm},
            follower_arms={"main": follower_arm},
            calibration_dir=self.calibration_dir,
            cameras=cameras,
        )
        self.robot = ManipulatorRobot(cfg)
        self.logger.debug("ManipulatorRobot instance created.")
        return self.robot

    def initialize(self):
        """
        Full initialization pipeline:
          1) setup_motors
          2) setup_cameras
          3) build_robot
        Returns: ManipulatorRobot
        """
        leader_arm, follower_arm = self.setup_motors()
        cameras = self.setup_cameras()
        return self.build_robot(leader_arm, follower_arm, cameras)

    def connect(self):
        """
        Connect to all devices (motors, cameras) via the ManipulatorRobot API.
        """
        if self.robot is None:
            raise RuntimeError("Call .initialize() before .connect()")
        self.logger.debug("Connecting to robot hardware...")
        self.robot.connect()
        self.logger.debug("Robot connected.")


if __name__ == "__main__":
    # example CLI launcher for iterative testing
    import argparse

    parser = argparse.ArgumentParser(description="Init Koch manipulator")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--log", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    # your motor & camera maps can be loaded from YAML/JSON here
    LEADER_MOTORS = {
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    }
    FOLLOWER_MOTORS = {
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    }
    CAMERA_PARAMS = {
        "front":    (4, 30, 640, 480),
        "overhead": (6, 30, 640, 480),
    }

    init = KochRobotInitializer(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        leader_motors=LEADER_MOTORS,
        follower_motors=FOLLOWER_MOTORS,
        camera_params=CAMERA_PARAMS,
        calibration_dir=".cache/calibration/koch",
        enable_logging=args.log,
    )
    robot = init.initialize()
    init.connect()
