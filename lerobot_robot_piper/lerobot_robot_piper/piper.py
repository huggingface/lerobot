from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_piper import PiperConfig
from .piper_sdk_interface import PiperSDKInterface


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self._iface: PiperSDKInterface | None = PiperSDKInterface(port=config.can_interface)
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}

    @property
    def is_connected(self) -> bool:
        return (self._iface is not None) and all(cam.is_connected for cam in self.cameras.values())

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        ft = {**self._motors_ft, **self._cameras_ft}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def action_features(self) -> dict:
        ft = {**self._motors_ft}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    def connect(self, calibrate: bool = True) -> None:
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def disconnect(self) -> None:
        if self._iface is not None:
            # Optionally perform a safe resume/stop here
            self._iface = None
        for cam in self.cameras.values():
            cam.disconnect()

    def is_calibrated(self) -> bool:  # type: ignore[override]
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _apply_signs(self, joints_deg: list[float]) -> list[float]:
        signs = self.config.joint_signs
        return [d * s for d, s in zip(joints_deg, signs, strict=True)]

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self._iface is None:
            raise ConnectionError(f"{self} is not connected.")
        status = self._iface.get_status_deg()
        # Gather in configured order and apply signs
        joints = [status[f"joint_{i+1}.pos"] for i in range(6)]
        joints = self._apply_signs(joints)
        obs = {name: val for name, val in zip(self.config.joint_names, joints, strict=True)}
        if self.config.include_gripper and "gripper.pos" in status:
            obs["gripper.pos"] = status["gripper.pos"]
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self._iface is None:
            raise ConnectionError(f"{self} is not connected.")
        joints = [float(action[f"{name}.pos"]) for name in self.config.joint_names]
        joints_hw = self._apply_signs(joints)
        gripper_mm = float(action["gripper.pos"]) if self.config.include_gripper and "gripper.pos" in action else None
        self._iface.set_joint_positions_deg(joints_hw, gripper_mm)
        return action