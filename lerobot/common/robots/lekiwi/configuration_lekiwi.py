from dataclasses import dataclass, field

from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.config import RobotConfig


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):

    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index="/dev/video0", fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"
    
    port_motor_bus = "/dev/ttyACM0"

    # TODO(Steven): consider split this into arm and base
    shoulder_pan: tuple = (1, "sts3215")
    shoulder_lift: tuple = (2, "sts3215")
    elbow_flex: tuple=(3, "sts3215")
    wrist_flex: tuple=(4, "sts3215")
    wrist_roll: tuple=(5, "sts3215")
    gripper: tuple =(6, "sts3215")
    left_wheel: tuple= (7, "sts3215")
    back_wheel: tuple =  (8, "sts3215")
    right_wheel: tuple = (9, "sts3215")
