from typing import Any
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot


class PiperArm(Robot):
    config_class = PiperArmConfig
    name = "piper_arm"

    def __init__(self, config: PiperArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)


    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def observation_features(self) -> dict[str, type]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft
    
    @property
    def connect(self) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()

    @property
    def disconnect(self) -> None:
        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
    
    def configure(self) -> None:
        self.bus.configure()
        for cam in self.cameras.values():
            cam.configure()
    
    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}
        for cam in self.cameras.values():
            obs_dict[cam.name] = cam.get_observation()
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: