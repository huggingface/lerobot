# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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
from functools import cached_property
from typing import Any
import numpy as np
import threading
import time

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController

from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import SourcceyProtobuf
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower
from .config_sourccey import SourcceyConfig

logger = logging.getLogger(__name__)

class Sourccey(Robot):
    """
    The robot includes a four mecanum wheel mobile base, 1 DC actuator, and 2 remote follower arms.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = SourcceyConfig
    name = "sourccey"

    def __init__(self, config: SourcceyConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SourcceyFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            motor_models=config.left_arm_motor_models,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )
        right_arm_config = SourcceyFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            motor_models=config.right_arm_motor_models,
            port=config.right_arm_port,
            orientation="right",
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SourcceyFollower(left_arm_config)
        self.right_arm = SourcceyFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

        self.dc_motors_controller = PWMDCMotorsController(
            motors=self.config.dc_motors,
            config=self.config.dc_motors_config,
        )

        # Initialize protobuf converter
        self.protobuf_converter = SourcceyProtobuf()

        # Track per-arm untorque state for edge detection
        self.untorque_left_prev = False
        self.untorque_right_prev = False

    def __del__(self):
        self.disconnect()

    ###################################################################
    # Properties and Attributes
    ###################################################################
    @property
    def _state_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"{motor}.pos": float for motor in self.right_arm.bus.motors} | {
                "x.vel": float,
                "y.vel": float,
                "z.vel": float,
                "theta.vel": float,
            }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        arms_connected = self.left_arm.is_connected and self.right_arm.is_connected
        cams_connected = all(self.cameras[k].is_connected for k in self.cameras.keys())
        return arms_connected and cams_connected

    ###################################################################
    # Connection Management
    ###################################################################
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        self.dc_motors_controller.connect()

        # Connect only target cameras
        for cam_key in self.cameras.keys():
            self.cameras[cam_key].connect()

    def disconnect(self):
        print("Disconnecting Sourccey")
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        self.stop_base()
        self.dc_motors_controller.disconnect()

        # Disconnect only those we connected
        for cam_key in self.cameras.keys():
            self.cameras[cam_key].disconnect()

    ###################################################################
    # Calibration and Configuration Management
    ###################################################################
    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def auto_calibrate(self, full_reset: bool = False, arm: str | None = None) -> None:
        """
        Auto-calibrate arms. If arm is None, calibrate both in parallel.
        arm can be "left" or "right" to calibrate only that side.
        """
        if arm is None:
            # Create threads for each arm
            left_thread = threading.Thread(
                target=self.left_arm.auto_calibrate,
                kwargs={"reversed": False, "full_reset": full_reset}
            )
            right_thread = threading.Thread(
                target=self.right_arm.auto_calibrate,
                kwargs={"reversed": True, "full_reset": full_reset}
            )

            # Start left arm immediately
            left_thread.start()

            # Wait 3 seconds before starting right arm
            time.sleep(3)
            right_thread.start()

            # Wait for both threads to complete
            left_thread.join()
            right_thread.join()
            return

        if arm not in ("left", "right"):
            raise ValueError("arm must be one of: None, 'left', 'right'")

        if arm == "left":
            self.left_arm.auto_calibrate(reversed=False, full_reset=full_reset)
        else:
            self.right_arm.auto_calibrate(reversed=True, full_reset=full_reset)

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    ###################################################################
    # Data Management
    ###################################################################

    def get_observation(self) -> dict[str, Any]:
        try:
            obs_dict = {}

            left_obs = self.left_arm.get_observation()
            obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

            right_obs = self.right_arm.get_observation()
            obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

            base_wheel_vel = self.dc_motors_controller.get_velocities()
            base_vel = self._wheel_normalized_to_body(base_wheel_vel)
            obs_dict.update(base_vel)

            for cam_key in self.cameras.keys():
                obs_dict[cam_key] = self.cameras[cam_key].async_read()

            return obs_dict
        except Exception as e:
            print(f"Error getting observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            # Apply per-arm untorque flags automatically
            action = self.apply_untorque_flags(action)

            left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
            right_action = {key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")}
            base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

            prefixed_send_action_left = {}
            prefixed_send_action_right = {}

            # Only send to followers if there are keys for that arm
            if left_action:
                sent_left = self.left_arm.send_action(left_action)
            else:
                sent_left = {}
            if right_action:
                sent_right = self.right_arm.send_action(right_action)
            else:
                sent_right = {}

            prefixed_send_action_left = {f"left_{key}": value for key, value in sent_left.items()}
            prefixed_send_action_right = {f"right_{key}": value for key, value in sent_right.items()}

            # Base velocity
            wheel_action = self._body_to_wheel_normalized(
                base_goal_vel.get("x.vel", 0.0),
                base_goal_vel.get("y.vel", 0.0),
                base_goal_vel.get("theta.vel", 0.0)
            )

            linear_actuator_action = self._body_to_linear_actuator_normalized(
                base_goal_vel.get("z.vel", 0.0)
            )

            dc_motors_action = {**wheel_action, **linear_actuator_action}
            self.dc_motors_controller.set_velocities(dc_motors_action)

            sent_action = {**prefixed_send_action_left, **prefixed_send_action_right, **base_goal_vel}
            return sent_action
        except Exception as e:
            print(f"Error sending action: {e}")
            return {}

    ###################################################################
    # Control Management
    ###################################################################
    def apply_untorque_flags(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Apply per-arm untorque flags: disable/enable torque and strip positions.
        Manages internal state for edge detection.

        Returns:
            dict: modified action with positions stripped if untorqued
        """
        left_flag = bool(action.get("untorque_left", False))
        right_flag = bool(action.get("untorque_right", False))

        # Left arm handling
        if left_flag:
            if not self.untorque_left_prev:
                self.left_arm.bus.disable_torque()
            action = {k: v for k, v in action.items() if not k.startswith("left_")}
        elif self.untorque_left_prev and not left_flag:
            self.left_arm.bus.enable_torque()

        # Right arm handling
        if right_flag:
            if not self.untorque_right_prev:
                self.right_arm.bus.disable_torque()
            action = {k: v for k, v in action.items() if not k.startswith("right_")}
        elif self.untorque_right_prev and not right_flag:
            self.right_arm.bus.enable_torque()

        # Update state
        self.untorque_left_prev = left_flag
        self.untorque_right_prev = right_flag

        return action

    def update(self):
        # Can be used to update the robot every cycle. Such as potentially a motor
        # self.dc_motors_controller.update_velocity(max_step=0.1)
        pass

    # Base Functions
    def stop_base(self):
        self.dc_motors_controller.set_velocities({"front_left": 0, "front_right": 0, "rear_left": 0, "rear_right": 0})

    ##################################################################################
    # Private Kinematic Functions
    ##################################################################################
    def _body_to_wheel_normalized(
        self,
        x: float,
        y: float,
        theta: float,
    ) -> dict:
        velocity_vector = np.array([x, y, theta])

        # Build the correct kinematic matrix for mecanum wheels
        # Flip the sign of the lateral (y) column to correct strafing direction
        m = np.array([
            [ 1, -1, -1], # Front-left wheel
            [-1, -1, -1], # Front-right wheel
            [ 1,  1, -1], # Rear-left wheel
            [-1,  1, -1], # Rear-right wheel
        ])

        wheel_normalized = m.dot(velocity_vector)
        wheel_normalized = np.clip(wheel_normalized, -1.0, 1.0)
        wheel_dict = {
            "front_left": float(wheel_normalized[0]),
            "front_right": float(wheel_normalized[1]),
            "rear_left": float(wheel_normalized[2]),
            "rear_right": float(wheel_normalized[3]),
        }

        return wheel_dict

    def _wheel_normalized_to_body(
        self,
        wheel_normalized: dict[str, Any],
    ) -> dict[str, Any]:

        # Convert each normalized command back to an angular speed in deg/s.
        wheel_array = np.array([
            wheel_normalized["front_left"],
            wheel_normalized["front_right"],
            wheel_normalized["rear_left"],
            wheel_normalized["rear_right"],
        ])

        # Kinematic matrix for mecanum wheels (must match forward kinematics)
        m = np.array([
            [ 1, -1, -1], # Front-left wheel
            [-1, -1, -1], # Front-right wheel
            [ 1,  1, -1], # Rear-left wheel
            [-1,  1, -1], # Rear-right wheel
        ])

        # Solve the inverse kinematics: body_velocity = M⁺ · wheel_linear_speeds.
        m_pinv = np.linalg.pinv(m)
        velocity_vector = m_pinv.dot(wheel_array)
        x, y, theta = velocity_vector

        return {
            "x.vel": self.clean_value(x),
            "y.vel": self.clean_value(y),
            "theta.vel": self.clean_value(theta),
        }

    def _body_to_linear_actuator_normalized(
        self,
        z: float,
    ) -> dict:
        return {
            "linear_actuator": self.clean_value(z),
        }

    def _linear_actuator_normalized_to_body(
        self,
        linear_actuator_normalized: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "z.vel": self.clean_value(linear_actuator_normalized["linear_actuator"]),
        }

    # Round to prevent floating-point precision issues and handle -0.0
    def clean_value(self, val):
        rounded = round(val, 8)

        # Convert -0.0 to 0.0 and very small values to 0.0
        return 0.0 if abs(rounded) < 1e-10 else rounded

    ##################################################################################
    # Motor Configuration Functions
    ##################################################################################
    def set_baud_rate(self, baud_rate: int) -> None:
        self.left_arm.bus.set_baudrate(baud_rate)
        self.right_arm.bus.set_baudrate(baud_rate)
