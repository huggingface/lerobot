#!/usr/bin/env python

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

import json
import logging
import time
import threading
import numpy as np
import time
# import torch
import base64
import cv2

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .configuration_lekiwi import LeKiwiRobotConfig
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    TorqueMode,
    run_arm_manual_calibration,
)
import zmq

class LeKiwiRobot(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = LeKiwiRobotConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiRobotConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id

        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        # TODO(Steven): Consider in the future using S100 robot class
        # TODO(Steven): Another option is to use the motorbus factory, but in this case we assume that
        # what we consider 'lekiwi robot' always uses the FeetechMotorsBus
        # TODO(Steven): We will need to have a key for arm and base for calibration
        self.actuators = FeetechMotorsBus(
            port=self.config.port_motor_bus,
            motors={
                "shoulder_pan": config.shoulder_pan,
                "shoulder_lift": config.shoulder_lift,
                "elbow_flex": config.elbow_flex,
                "wrist_flex": config.wrist_flex,
                "wrist_roll": config.wrist_roll,
                "gripper": config.gripper,
                "left_wheel": config.left_wheel,
                "right_wheel": config.right_wheel,
                "back_wheel": config.back_wheel,
            },
        )

        #TODO(Steven): Consider removing cameras from configs
        self.cameras = make_cameras_from_configs(config.cameras)        

        self.observation_lock = threading.Lock()
        self.last_observation = None

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.is_connected = False
        self.logs = {}

        

    @property
    def state_feature(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.actuators),),
            "names": {"motors": list(self.actuators.motors)},
        }

    @property
    def action_feature(self) -> dict:
        return self.state_feature

    @property
    def camera_features(self) -> dict[str, dict]:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft[cam_key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    def setup_zmq_sockets(self):
        context = zmq.Context()
        cmd_socket = context.socket(zmq.PULL)
        cmd_socket.setsockopt(zmq.CONFLATE, 1)
        cmd_socket.bind(f"tcp://*:{self.port_zmq_cmd}")

        observation_socket = context.socket(zmq.PUSH)
        observation_socket.setsockopt(zmq.CONFLATE, 1)
        observation_socket.bind(f"tcp://*:{self.port_zmq_observations}")

        return context, cmd_socket, observation_socket
    
    def setup_actuators(self):
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.actuators.write("Torque_Enable", TorqueMode.DISABLED.value)
        self.calibrate()

        # Mode=0 for Position Control
        # TODO(Steven): Base robots should actually be in vel mode
        self.actuators.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        self.actuators.write("P_Coefficient", 16)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.actuators.write("I_Coefficient", 0)
        self.actuators.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.actuators.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.actuators.write("Maximum_Acceleration", 254)
        self.actuators.write("Acceleration", 254)

        logging.info("Activating torque.")
        self.actuators.write("Torque_Enable", TorqueMode.ENABLED.value)

        # Check arm can be read
        self.actuators.read("Present_Position")

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "LeKiwi Robot is already connected. Do not run `robot.connect()` twice."
            )

        logging.info("Connecting actuators.")
        self.actuators.connect()
        self.setup_actuators()

        logging.info("Connecting cameras.")
        for cam in self.cameras.values():
            cam.connect()

        logging.info("Connecting ZMQ sockets.")
        self.zmq_context, self.zmq_cmd_socket, self.zmq_observation_socket = self.setup_zmq_sockets(self.config)

        self.is_connected = True

    # TODO(Steven): Consider using this
    # def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
    #     return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    def calibrate(self) -> None:
        # Copied from S100 robot
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        actuators_calib_path = self.calibration_dir / f"{self.config.id}.json"

        if actuators_calib_path.exists():
            with open(actuators_calib_path) as f:
                calibration = json.load(f)
        else:
            logging.info(f"Missing calibration file '{actuators_calib_path}'")
            calibration = run_arm_manual_calibration(self.actuators, self.robot_type, self.name, "follower")

            logging.info(f"Calibration is done! Saving calibration file '{actuators_calib_path}'")
            actuators_calib_path.parent.mkdir(parents=True, exist_ok=True)
            with open(actuators_calib_path, "w") as f:
                json.dump(calibration, f)

        self.actuators.set_calibration(calibration)

    def get_observation(self) -> dict[str, np.ndarray]:
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "LeKiwiRobot is not connected. You need to run `robot.connect()`."
            )

        obs_dict = {}

        # Read actuators position
        # TODO(Steven): Base motors should return a vel instead of a pos
        before_read_t = time.perf_counter()
        obs_dict[OBS_STATE] = self.actuators.read("Present_Position")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            frame = cam.async_read()
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ret:
                obs_dict[f"{OBS_IMAGES}.{cam_key}"] = base64.b64encode(buffer).decode("utf-8")
            else:
                obs_dict[f"{OBS_IMAGES}.{cam_key}"] = ""
            self.logs[f"read_camera_{cam_key}_dt_s"] = cam.logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:
        # Copied from S100 robot
        """Command lekiwi to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (np.ndarray): array containing the goal positions for the motors.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "LeKiwiRobot is not connected. You need to run `robot.connect()`."
            )

        goal_pos = action

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.actuators.read("Present_Position")
            goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

        # Send goal position to the actuators
        # TODO(Steven): Base motors should set a vel instead
        self.actuators.write("Goal_Position", goal_pos.astype(np.int32))

        return goal_pos

    def update_last_observation(self, stop_event):
        while not stop_event.is_set():
            obs = self.get_observation()
            with self.observation_lock:
                self.last_observation = obs
            # TODO(Steven): Consider adding a delay to not starve the CPU

    def stop(self):
        # TODO(Steven): Base motors speed should be set to 0
        pass

    def run(self):
        # Copied logic from run_lekiwi in lekiwi_remote.py
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "LeKiwiRobot is not connected. You need to run `robot.connect()`."
            )
        
        stop_event = threading.Event()
        observation_thread = threading.Thread(
            target=self.update_last_observation, args=(stop_event), daemon=True
        )
        observation_thread.start()

        last_cmd_time = time.time()
        logging.info("LeKiwi robot server started. Waiting for commands...")

        try:
            while True:
                loop_start_time = time.time()

                try:
                    msg = self.cmd_socket.recv_string(zmq.NOBLOCK)
                    data = json.loads(msg)
                    # TODO(Steven): Process data correctly
                    self.send_action(data)
                    last_cmd_time = time.time()
                # except zmq.Again:
                #     logging.warning("ZMQ again")
                except Exception as e:
                    logging.warning(f"[ERROR] Message fetching failed: {e}")
                
                # Watchdog: stop the robot if no command is received for over 0.5 seconds.
                now = time.time()
                if now - last_cmd_time > 0.5:
                    self.stop() 
                    pass
                
                with self.observation_lock:
                    self.zmq_observation_socket.send_string(json.dumps(self.last_observation))
                
                # Ensure a short sleep to avoid overloading the CPU.
                elapsed = time.time() - loop_start_time
                time.sleep(
                    max(0.033 - elapsed, 0)
                )  # If robot jitters increase the sleep and monitor cpu load with `top` in cmd
        except KeyboardInterrupt:
            print("Shutting down LeKiwi server.")
        finally:
            stop_event.set()
            observation_thread.join()
            self.disconnect()
            pass
    
    def print_logs(self):
        # TODO(Steven): Refactor logger
        pass

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "LeKiwi is not connected. You need to run `robot.connect()` before disconnecting."
            )
        
        self.stop()
        self.actuators.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        self.observation_socket.close()
        self.cmd_socket.close()
        self.context.term()
        self.is_connected = False
    
    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
