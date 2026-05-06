# #!/usr/bin/env python

# # Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import logging

# import onerogripper_api_py as onerogripper

# from lerobot.teleoperators.teleoperator import Teleoperator
# from lerobot.utils.errors import DeviceNotConnectedError

# from ...robots.onerobotics_follower.onero_arm import OneroAdapter
# from .config_onero_arm_teleoperate import OneroTeleopLeaderConfig

# logger = logging.getLogger("onero_arm")


# class OneroTeleopLeader(Teleoperator):
#     """
#     Teleoperator wrapper for the Onero leader arm.

#     This class implements the `Teleoperator` interface and delegates low-level
#     robot interactions to a `OneroAdapter` instance. Using composition keeps
#     the teleoperator visible to the rest of the system as a `Teleoperator`.
#     """

#     config_class = OneroTeleopLeaderConfig
#     name = "onero_teleop_leader"

#     def __init__(self, config: OneroTeleopLeaderConfig):
#         super().__init__(config)
#         self.config = config
#         self._adapter = OneroAdapter(config)
#         self._gripper_joystick = None

#     @property
#     def action_features(self) -> dict[str, type]:
#         return self._adapter.action_features

#     @property
#     def feedback_features(self) -> dict[str, type]:
#         return {}

#     @property
#     def is_connected(self) -> bool:
#         return self._adapter.is_connected

#     def connect(self, calibrate: bool = True) -> None:
#         self._adapter.connect()
#         # Start gravity compensation if available
#         try:
#             if self._adapter.is_connected:
#                 # Move main arm to predefined safe joints before enabling gravity compensation
#                 home_joints_positions = getattr(self.config, "home_joints_positions", None)
#                 if home_joints_positions is not None:
#                     self._adapter._arm.movej(home_joints_positions, speed_scale=0.8, trajectory_connect=0)

#                 self._adapter._arm.ArmGravityCompensation()
#                 logger.info(f"{self} gravity compensation started.")
#                 if self.config.enable_gripper_joystick:
#                     self._gripper_joystick = onerogripper.GripperControl()
#                     if not self._gripper_joystick.initialize_joystick(
#                         self.config.port, self.config.slcan_type
#                     ):
#                         logger.warning("Failed to connect to gripper joystick control.")
#         except Exception:
#             logger.exception("Error during leader connect/gravity compensation")

#     @property
#     def is_calibrated(self) -> bool:
#         # Delegate to adapter when applicable
#         return getattr(self._adapter, "is_calibrated", True)

#     def calibrate(self) -> None:
#         # No-op: adapter handles any hardware calibration
#         return None

#     def configure(self) -> None:
#         # No special configuration beyond adapter
#         return None

#     def get_action(self):
#         if not self._adapter.is_connected:
#             raise DeviceNotConnectedError(f"{self} is not connected.")

#         obs = self._adapter.get_observation()
#         if self.config.enable_gripper_joystick and self._gripper_joystick is not None:
#             try:
#                 obs["gripper.position"] = self._gripper_joystick.get_joystick_pos()
#             except Exception:
#                 logger.exception("Failed reading gripper joystick position")

#         return obs

#     def send_feedback(self, feedback: dict[str, any]) -> None:
#         # Leader teleop typically doesn't accept feedback; no-op for now
#         return None

#     def disconnect(self) -> None:
#         try:
#             if self._gripper_joystick:
#                 # If the gripper joystick has a cleanup method, call it.
#                 self._gripper_joystick = None
#         finally:
#             self._adapter.disconnect()


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
import threading
import time

import onerogripper_api_py as onerogripper

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError

from ...robots.onerobotics_follower.onero_arm import OneroAdapter
from .config_onero_arm_teleoperate import OneroTeleopLeaderConfig

logger = logging.getLogger("onero_arm")


class OneroTeleopLeader(Teleoperator):
    """
    Teleoperator wrapper for the Onero leader arm.

    This class implements the `Teleoperator` interface and delegates low-level
    robot interactions to a `OneroAdapter` instance. Using composition keeps
    the teleoperator visible to the rest of the system as a `Teleoperator`.
    """

    config_class = OneroTeleopLeaderConfig
    name = "onero_teleop_leader"

    def __init__(self, config: OneroTeleopLeaderConfig):
        super().__init__(config)
        self.config = config
        self._adapter = OneroAdapter(config)
        self._gripper_joystick = None

        self._reader_thread = None
        self._reader_running = False
        self._action_lock = threading.Lock()
        self._latest_action = None

        self._latest_action = None
        self._latest_action_ts = None
        self._latest_action_age_ms = None

        # 保守轮询频率，避免抢占底层总线/SDK
        self._cache_hz = 20.0

    @property
    def action_features(self) -> dict[str, type]:
        return self._adapter.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._adapter.is_connected

    @property
    def latest_action_age_ms(self):
        return self._latest_action_age_ms
    
    def _read_action_once(self):
        obs = self._adapter.get_observation()
        if self.config.enable_gripper_joystick and self._gripper_joystick is not None:
            try:
                obs["gripper.position"] = self._gripper_joystick.get_joystick_pos()
            except Exception:
                logger.exception("Failed reading gripper joystick position")
        return obs

    def _reader_loop(self):
        period = 1.0 / self._cache_hz
        while self._reader_running:
            t0 = time.perf_counter()
            try:
                if self._adapter.is_connected:
                    obs = self._read_action_once()
                    with self._action_lock:
                        self._latest_action = obs
                        self._latest_action_ts = time.perf_counter()
            except Exception:
                logger.exception("Failed updating cached leader action")

            elapsed = time.perf_counter() - t0
            sleep_time = max(period - elapsed, 0.005)
            time.sleep(sleep_time)

    def connect(self, calibrate: bool = True) -> None:
        self._adapter.connect()
        # Start gravity compensation if available
        try:
            if self._adapter.is_connected:
                # Move main arm to predefined safe joints before enabling gravity compensation
                home_joints_positions = getattr(self.config, "home_joints_positions", None)
                if home_joints_positions is not None:
                    self._adapter._arm.movej(home_joints_positions, speed_scale=0.8, trajectory_connect=0)

                self._adapter._arm.ArmGravityCompensation()
                logger.info(f"{self} gravity compensation started.")
                if self.config.enable_gripper_joystick:
                    self._gripper_joystick = onerogripper.GripperControl()
                    if not self._gripper_joystick.initialize_joystick(
                        self.config.port, self.config.slcan_type
                    ):
                        logger.warning("Failed to connect to gripper joystick control.")
        except Exception:
            logger.exception("Error during leader connect/gravity compensation")

        self._reader_running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    @property
    def is_calibrated(self) -> bool:
        # Delegate to adapter when applicable
        return getattr(self._adapter, "is_calibrated", True)

    def calibrate(self) -> None:
        # No-op: adapter handles any hardware calibration
        return None

    def configure(self) -> None:
        # No special configuration beyond adapter
        return None

    # def get_action(self):
    #     if not self._adapter.is_connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")

    #     with self._action_lock:
    #         cached = self._latest_action

    #     if cached is not None:
    #         return cached

    #     # 仅在刚启动缓存尚未准备好时回退一次同步读取
    #     return self._read_action_once()

    def get_action(self):
        if not self._adapter.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._action_lock:
            cached = self._latest_action
            cached_ts = self._latest_action_ts

        if cached is not None:
            self._latest_action_age_ms = (
                (time.perf_counter() - cached_ts) * 1000 if cached_ts is not None else None
            )
            return cached

        obs = self._read_action_once()
        self._latest_action_age_ms = 0.0
        return obs


    def send_feedback(self, feedback: dict[str, any]) -> None:
        # Leader teleop typically doesn't accept feedback; no-op for now
        return None

    def disconnect(self) -> None:
        try:
            self._reader_running = False
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=1.0)
                self._reader_thread = None

            if self._gripper_joystick:
                self._gripper_joystick = None
        finally:
            self._adapter.disconnect()