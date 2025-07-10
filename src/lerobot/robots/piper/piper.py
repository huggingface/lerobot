# Implementation of Piper robot for LeRobot

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from typing import Any
from .piper_sdk_interface import PiperSDKInterface

@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.sdk = PiperSDKInterface(port=config.port)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"joint_{i}.pos": float for i in range(7)
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        # Assume always connected after SDK init
        return True

    def connect(self, calibrate: bool = True) -> None:
        # Already connected in SDK init
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def disconnect(self) -> None:
        self.sdk.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        obs_dict = self.sdk.get_status()
        
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # map the action from the leader to joints for the follower
        positions = [ 
            action.get("shoulder_pan.pos"),
            action.get("shoulder_lift.pos"),
            action.get("elbow_flex.pos"),
            0,
            action.get("wrist_flex.pos"),
            action.get("wrist_roll.pos"),
            action.get("gripper.pos"),
        ]
        
        self.sdk.set_joint_positions(positions)
        return action
