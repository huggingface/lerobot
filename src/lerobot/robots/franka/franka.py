from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots import Robot
from .franka_config import FrankaConfig
from openteach.utils.network import ZMQCameraSubscriber
from openteach.components.operators.franka import (
    CONFIG_ROOT,
    CONTROL_FREQ,
    ROTATION_VELOCITY_LIMIT,
    STATE_FREQ,
    TRANSLATION_VELOCITY_LIMIT,
    FrankaArmOperator,
)
import yaml
import os
from easydict import EasyDict

class FrankaRobot(Robot):
    config_class = FrankaConfig
    name = "franka"

    def __init__(self, config: FrankaConfig):
        super().__init__(config)
        # TODO verify the cameras are named corectly
        self.front_subscriber = ZMQCameraSubscriber(
                host = "172.16.0.1",
                port = "10007",
                topic_type = 'RGB'
            )

        self.wrist_subscriber = ZMQCameraSubscriber(
                host = "172.16.0.1",
                port = "10006",
                topic_type = 'RGB'
            )

        self.side_subscriber = ZMQCameraSubscriber(
                host = "172.16.0.1",
                port = "10005",
                topic_type = 'RGB'
            )
        with open(os.path.join(CONFIG_ROOT, "network.yaml"), "r") as f:
            network_cfg = EasyDict(yaml.safe_load(f))
        self.operator = FrankaArmOperator(
            network_cfg["host_address"],
            None,
            None,
            None,
            use_filter=False,
            arm_resolution_port = None,
            teleoperation_reset_port = None,
            record='test_lerobot',
            )
        self.cameras = [
            self.front_subscriber,
            self.wrist_subscriber,
            self.side_subscriber,
            ]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            # "joint_1.pos": float,
            # "joint_2.pos": float,
            # "joint_3.pos": float,
            # "joint_4.pos": float,
            # "joint_5.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            # cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    def configure(self) -> None:
        pass

    def send_action(self, action) -> None:
        arm_action = action[:6]
        gripper_action = action[:-1]
        playback_actions = (arm_action, gripper_action)
        self.operator.arm_control(None, None, playback_actions=playback_actions)

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        pass

    def disconnect(self) -> None:
        pass


    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    # TODO should any normalization or preprocessing be done here?
    def get_observation(self):
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = {}
        obs_dict["front_cam"] = self.front_subscriber.recv_rgb_image()
        obs_dict["side_cam"] = self.side_subscriber.recv_rgb_image()
        obs_dict["wrist_cam"] = self.wrist_subscriber.recv_rgb_image()
        obs_dict["joint_pos"] = self.operator.robot_interface.last_q.tolist()
        obs_dict["gripper_pos"] = self.operator.robot_interface.last_gripper_q
        return obs_dict