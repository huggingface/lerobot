# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import base64
import json
import logging
from typing import Any

import cv2
import numpy as np
import torch
import zmq

from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError, InvalidActionError

from ..robot import Robot
from .config_lekiwi import LeKiwiClientConfig, RobotMode


# TODO(Steven): This doesn't need to inherit from Robot
# But we do it for now to offer a familiar API
# TODO(Steven): This doesn't need to take care of the
# mapping from teleop to motor commands, but given that
# we already have a middle-man (this class) we add it here
# Other options include:
# 1. Adding it to the Telop implementation for lekiwi
# (meaning each robot will need a teleop imple) or
# 2. Adding it into the robot implementation
# (meaning the policy might be needed to be train
# over the teleop action space)
# TODO(Steven): Check if we can move everything to 32 instead
class LeKiwiClient(Robot):
    config_class = LeKiwiClientConfig
    name = "lekiwi_client"

    def __init__(self, config: LeKiwiClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type
        self.robot_mode = config.robot_mode

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames = {}
        self.last_present_speed = {"x_cmd": 0.0, "y_cmd": 0.0, "theta_cmd": 0.0}

        self.last_remote_arm_state = {}

        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow

        self._is_connected = False
        self.logs = {}

    @property
    def state_feature(self) -> dict:
        # TODO(Steven): Get this from the data fetched? Motor names are unknown for the Daemon
        # For now we assume its size/metadata is known
        return {
            "dtype": "float64",
            "shape": (9,),
            "names": {
                "motors": [
                    "arm_shoulder_pan",
                    "arm_shoulder_lift",
                    "arm_elbow_flex",
                    "arm_wrist_flex",
                    "arm_wrist_roll",
                    "arm_gripper",
                    "base_left_wheel",
                    "base_right_wheel",
                    "base_back_wheel",
                ]
            },
        }

    @property
    def action_feature(self) -> dict:
        return self.state_feature

    @property
    def camera_features(self) -> dict[str, dict]:
        # TODO(Steven): Get this from the data fetched? Motor names are unknown for the Daemon
        # For now we assume its size/metadata is known
        # TODO(Steven): Check consistency of image sizes
        cam_ft = {
            "front": {
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            },
            "wrist": {
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            },
        }
        return cam_ft

    @property
    def is_connected(self) -> bool:
        # TODO(Steven): Ideally we could check instead the status of the sockets
        # I didn't find any API that allows us to do that easily
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        """Establishes ZMQ sockets with the remote mobile robot"""

        # TODO(Steven): Consider instead returning a bool + warn
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "LeKiwi Daemon is already connected. Do not run `robot.connect()` twice."
            )

        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        self._is_connected = True

    def calibrate(self) -> None:
        logging.warning("LeKiwiClient has nothing to calibrate.")
        return

    # Consider moving these static functions out of the class
    # Copied from robot_lekiwi MobileManipulator class* (before the refactor)
    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    # Copied from robot_lekiwi MobileManipulator class* (before the refactor)
    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    # Copied from robot_lekiwi MobileManipulator class* (before the refactor)
    def _body_to_wheel_raw(
        self,
        x_cmd: float,
        y_cmd: float,
        theta_cmd: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"left_wheel": value, "back_wheel": value, "right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta_cmd * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 120, 0]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [LeKiwiClient._degps_to_raw(deg) for deg in wheel_degps]

        # TODO(Steven): remove hard-coded names
        return {"left_wheel": wheel_raw[0], "back_wheel": wheel_raw[1], "right_wheel": wheel_raw[2]}

    # Copied from robot_lekiwi MobileManipulator class
    def _wheel_raw_to_body(
        self, wheel_raw: dict[str, Any], wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("left_wheel", "back_wheel", "right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A tuple (x_cmd, y_cmd, theta_cmd) where:
             x_cmd      : Linear velocity in x (m/s).
             y_cmd      : Linear velocity in y (m/s).
             theta_cmd  : Rotational velocity in deg/s.
        """

        # TODO(Steven): No check is done for dict keys
        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([LeKiwiClient._raw_to_degps(int(v)) for _, v in wheel_raw.items()])
        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 120, 0]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x_cmd, y_cmd, theta_rad = velocity_vector
        theta_cmd = theta_rad * (180.0 / np.pi)
        return {"x_cmd": x_cmd, "y_cmd": y_cmd, "theta_cmd": theta_cmd}

    # TODO(Steven): This is flaky, for example, if we received a state but failed decoding the image, we will not update any value
    # TODO(Steven): All this function needs to be refactored
    # Copied from robot_lekiwi MobileManipulator class* (before the refactor)
    def _get_data(self):
        # Copied from robot_lekiwi.py
        """Polls the video socket for up to 15 ms. If data arrives, decode only
        the *latest* message, returning frames, speed, and arm state. If
        nothing arrives for any field, use the last known values."""

        frames = {}
        present_speed = {}

        remote_arm_state_tensor = {}

        # Poll up to 15 ms
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(15))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            # No new data arrived → reuse ALL old data
            # TODO(Steven): This might return empty variables at init
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Drain all messages, keep only the last
        last_msg = None
        # TODO(Steven): There's probably a way to do this without while True
        # TODO(Steven): Even consider changing to PUB/SUB
        while True:
            try:
                obs_string = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = obs_string
            except zmq.Again:
                break

        if not last_msg:
            # No new message → also reuse old
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Decode only the final message
        try:
            observation = json.loads(last_msg)

            state_observation = observation[OBS_STATE]
            image_observation = observation[OBS_IMAGES]

            # Convert images
            for cam_name, image_b64 in image_observation.items():
                if image_b64:
                    jpg_data = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                    frame_candidate = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_candidate is not None:
                        frames[cam_name] = frame_candidate

            # TODO(Steven): Should we really ignore the arm state if the image is None?
            # If remote_arm_state is None and frames is None there is no message then use the previous message
            if state_observation is not None and frames is not None:
                self.last_frames = frames

                # TODO(Steven): hard-coded name of expected keys, not good
                remote_arm_state_tensor = {k: v for k, v in state_observation.items() if k.startswith("arm")}
                self.last_remote_arm_state = remote_arm_state_tensor

                present_speed = {k: v for k, v in state_observation.items() if k.startswith("base")}
                self.last_present_speed = present_speed
            else:
                frames = self.last_frames
                remote_arm_state_tensor = self.last_remote_arm_state
                present_speed = self.last_present_speed

        except Exception as e:
            print(f"[DEBUG] Error decoding video message: {e}")
            # If decode fails, fall back to old data
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)
        return frames, present_speed, remote_arm_state_tensor

    # TODO(Steven): The returned space is different from the get_observation of LeKiwi
    # This returns body-frames velocities instead of wheel pos/speeds
    def get_observation(self) -> dict[str, Any]:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame. Receives over ZMQ, translate to body-frame vel
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("LeKiwiClient is not connected. You need to run `robot.connect()`.")

        # TODO(Steven): remove hard-coded cam names & dims
        # This is needed at init for when there's no comms
        obs_dict = {
            OBS_IMAGES: {"wrist": np.zeros(shape=(480, 640, 3)), "front": np.zeros(shape=(640, 480, 3))}
        }

        frames, present_speed, remote_arm_state_tensor = self._get_data()
        body_state = self._wheel_raw_to_body(present_speed)
        # TODO(Steven): output is dict[str,Any] and we multiply by 1000.0. This should be more explicit and specify the expected type instead of Any
        body_state_mm = {k: v * 1000.0 for k, v in body_state.items()}  # Convert x,y to mm/s

        obs_dict[OBS_STATE] = {**remote_arm_state_tensor, **body_state_mm}

        # Loop over each configured camera
        for cam_name, frame in frames.items():
            if frame is None:
                # TODO(Steven): Daemon doesn't know camera dimensions (hard-coded for now), consider at least getting them from state features
                logging.warning("Frame is None")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            obs_dict[OBS_IMAGES][cam_name] = torch.from_numpy(frame)

        return obs_dict

    def _from_keyboard_to_wheel_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
        return self._body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

    def configure(self):
        pass

    # TODO(Steven): This assumes this call is always called with a keyboard as a teleop device. It breaks if we teleop with other device
    # TODO(Steven): Doing this mapping in here adds latecy between send_action and movement from the user perspective.
    # t0: get teleop_cmd
    # t1: send_action(teleop_cmd)
    # t2: mapping teleop_cmd -> motor_cmd
    # t3: execute motor_md
    # This mapping for other robots/teleop devices might be slower. Doing this in the teleop will make this explicit
    # t0': get teleop_cmd
    # t1': mapping teleop_cmd -> motor_cmd
    # t2': send_action(motor_cmd)
    # t3': execute motor_cmd
    # t3'-t2' << t3-t1
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command lekiwi to move to a target joint configuration. Translates to motor space + sends over ZMQ

        Args:
            action (np.ndarray): array containing the goal positions for the motors.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        if self.robot_mode is RobotMode.AUTO:
            # TODO(Steven): Not yet implemented. The policy outputs might need a different conversion
            raise InvalidActionError("Policy output as action input is not yet well defined")

        goal_pos = {}
        # TODO(Steven): This assumes teleop mode is always used with keyboard. Tomorrow we could teleop with another device ... ?
        if self.robot_mode is RobotMode.TELEOP:
            motors_name = self.state_feature.get("names").get("motors")

            common_keys = [
                key for key in action if key in (motor.replace("arm_", "") for motor in motors_name)
            ]

            # TODO(Steven): I don't like this
            if len(common_keys) < 6:
                logging.error("Action should include at least the states of the leader arm")
                raise InvalidActionError

            arm_actions = {"arm_" + arm_motor: action[arm_motor] for arm_motor in common_keys}
            goal_pos = arm_actions

            if len(action) > 6:
                keyboard_keys = np.array(list(set(action.keys()) - set(common_keys)))
                wheel_actions = {
                    "base_" + k: v for k, v in self._from_keyboard_to_wheel_action(keyboard_keys).items()
                }
                goal_pos = {**arm_actions, **wheel_actions}

            self.zmq_cmd_socket.send_string(json.dumps(goal_pos))  # action is in motor space

        return goal_pos

    def print_logs(self):
        # TODO(Steven): Refactor logger
        pass

    def disconnect(self):
        """Cleans ZMQ comms"""

        if not self._is_connected:
            raise DeviceNotConnectedError(
                "LeKiwi is not connected. You need to run `robot.connect()` before disconnecting."
            )
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
