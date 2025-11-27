#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.trossen import TrossenArmDriver

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_widow_ai_follower import WidowAIFollowerConfig

logger = logging.getLogger(__name__)


class WidowAIFollower(Robot):
    config_class = WidowAIFollowerConfig
    name = "widow_ai_follower"

    def __init__(self, config: WidowAIFollowerConfig):
        super().__init__(config)
        self.config = config
        
        self.bus = TrossenArmDriver(
            port=self.config.port,
            model=self.config.model,
            velocity_limit_scale=self.config.velocity_limit_scale,
        )
        
        self.motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_1", "wrist_2", "wrist_3", "gripper"]
        
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _efforts_ft(self) -> dict[str, type]:
        return {f"{motor}.effort": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        if self.config.effort_sensing:
            return {**self._motors_ft, **self._efforts_ft, **self._cameras_ft}
        else:
            return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Trossen arms are pre-calibrated
        return True

    def calibrate(self) -> None:
        logger.info(f"{self} is pre-calibrated, no calibration needed.")

    def configure(self) -> None:
        self.bus.initialize_for_teleoperation(is_leader=False)

    def setup_motors(self) -> None:
        logger.info(f"{self} motors are pre-configured.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        positions = self.bus.read("Present_Position")
        
        # Convert numpy array to dict with motor names
        obs_dict = {}
        for i, motor in enumerate(self.motor_names):
            if i < len(positions):
                obs_dict[f"{motor}.pos"] = float(positions[i])
            else:
                obs_dict[f"{motor}.pos"] = 0.0
                
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        if self.config.effort_sensing:
            try:
                start = time.perf_counter()
                efforts = self.bus.read("External_Efforts")
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read external efforts: {dt_ms:.1f}ms")

                # Convert numpy array to dict with motor names
                for i, motor in enumerate(self.motor_names):
                    if i < len(efforts):
                        obs_dict[f"{motor}.effort"] = float(efforts[i])
                    else:
                        obs_dict[f"{motor}.effort"] = 0.0
            except Exception as e:
                logger.debug(f"{self} failed to read external efforts, using zeros: {e}")
                for motor in self.motor_names:
                    obs_dict[f"{motor}.effort"] = 0.0

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        
        # Extract goal positions from action dict
        goal_pos = []
        for motor in self.motor_names:
            pos_key = f"{motor}.pos"
            if pos_key in action:
                goal_pos.append(action[pos_key])
            else:
                goal_pos.append(0.0)

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.read("Present_Position")
            present_dict = {motor: float(present_pos[i]) for i, motor in enumerate(self.motor_names) if i < len(present_pos)}
            goal_dict = {motor: goal_pos[i] for i, motor in enumerate(self.motor_names)}
            goal_present_pos = {key: (goal_dict[key], present_dict.get(key, 0.0)) for key in goal_dict}
            safe_goal_dict = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            goal_pos = [safe_goal_dict.get(motor, 0.0) for motor in self.motor_names]

        self.bus.write("Goal_Position", goal_pos)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        
        return {f"{motor}.pos": val for motor, val in zip(self.motor_names, goal_pos)}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
