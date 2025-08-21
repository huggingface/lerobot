#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.robots.viperx import ViperX, ViperXConfig

from .config_aloha_follower import AlohaFollowerConfig

logger = logging.getLogger(__name__)


PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

def puppet_gripper_normalize(x): 
    return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
        PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    )

def puppet_gripper_unnormalize(x):
    return x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE


class AlohaFollower(Robot):
    config_class = AlohaFollowerConfig
    name = "aloha_follower"

    def __init__(self, config: AlohaFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = ViperXConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            max_relative_target=config.max_relative_target,
            cameras={},
        )

        right_arm_config = ViperXConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            max_relative_target=config.max_relative_target,
            cameras={},
        )

        self.left_arm = ViperX(left_arm_config)
        self.right_arm = ViperX(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}": float for motor in self.left_arm._motors_ft} | {
            f"right_{motor}": float for motor in self.right_arm._motors_ft
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = False) -> None:
        if calibrate:
            raise ValueError("Aloha robot does not support calibration on connect.")

        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        if self.config.normalized_gripper:
            obs_dict["left_finger.pos"] = puppet_gripper_normalize(obs_dict["left_finger.pos"])
            obs_dict["right_finger.pos"] = puppet_gripper_normalize(obs_dict["right_finger.pos"])

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        def process_arm(prefix):
            # Remove prefix
            arm_action = {
                key.removeprefix(f"{prefix}_"): value
                for key, value in action.items()
                if key.startswith(f"{prefix}_")
            }
            # Normalize gripper if needed
            if "finger.pos" in arm_action and self.config.normalized_gripper:
                arm_action["finger.pos"] = max(0, min(1.0, arm_action["finger.pos"]))
                arm_action["finger.pos"] = puppet_gripper_unnormalize(arm_action["finger.pos"])
            # Send action
            send_action = getattr(self, f"{prefix}_arm").send_action(arm_action)
            # Add prefix back
            return {f"{prefix}_{key}": value for key, value in send_action.items()}

        prefixed_send_action_left = process_arm("left")
        prefixed_send_action_right = process_arm("right")

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

    def send_home_cmd(self):
        start_pos = (0, -0.96, 1.16, 0, -0.3, 0, 0.02)
        joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "finger"]
        cmd = {name + ".pos": val for name, val in zip(joint_names, start_pos, strict=False)}
        left_cmd = {f"left_{k}": v for k, v in cmd.items()}
        right_cmd = {f"right_{k}": v for k, v in cmd.items()}
        self.send_action({**left_cmd, **right_cmd})

    # Move to a position just above the cradle to enable a safe torque disable
    def send_sleep_cmd(self):
        start_pos = (0, -1.8, 1.6, 0.12, 0.65, 0, 0.05)
        joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "finger"]
        cmd = {name + ".pos": val for name, val in zip(joint_names, start_pos, strict=False)}
        left_cmd = {f"left_{k}": v for k, v in cmd.items()}
        right_cmd = {f"right_{k}": v for k, v in cmd.items()}
        self.send_action({**left_cmd, **right_cmd})

    def reset_arms(self):
        for _ in range(300):
            self.send_home_cmd()
            time.sleep(0.01)

    def __enter__(self):
        if not self.is_connected:
            self.connect(calibrate=False)
        for _ in range(200):
            self.send_home_cmd()
            time.sleep(0.01)
        self.active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        while True:
            try:
                for _ in range(100):
                    self.send_home_cmd()
                    time.sleep(0.01)
                for _ in range(100):
                    self.send_sleep_cmd()
                    time.sleep(0.01)
                self.active = False
                self.disconnect()
                return
            except KeyboardInterrupt:
                print("Suppressing KeyboardInterrupt during cleanup.")
