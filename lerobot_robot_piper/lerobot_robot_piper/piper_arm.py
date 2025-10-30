from __future__ import annotations

from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_piper_arm import PiperArmConfig
from .piper_can_bus import PiperCanBus


class PiperArm(Robot):
    config_class = PiperArmConfig
    name = "piper_arm"

    def __init__(self, config: PiperArmConfig):
        super().__init__(config)
        self.bus = PiperCanBus(
            interface=config.can_interface,
            bitrate=config.bitrate,
            joint_names=config.joint_names,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def _motors_ft(self) -> dict[str, type]:
        # LeRobot schema: joint features by name as floats
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (c.height, c.width, 3) for cam, c in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        self.bus.disconnect()

    def configure(self) -> None:
        # Optional: send SDK configs (modes/gains) if needed
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        obs = self.bus.read_positions()
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goals = {k.removesuffix(".pos"): v for k, v in action.items()}
        self.bus.write_goal_positions(goals)
        return action