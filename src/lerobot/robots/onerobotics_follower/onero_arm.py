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


import copy
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import transforms3d as t3d

# Adapt imports to your environment.
# Assuming oneroarm_api_py is available in python path
import oneroarm_api_py as oneroarm
import onerogripper_api_py as onerogripper

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_onero_arm import OneroRobotConfig

logger = logging.getLogger("onero_arm")


class OneroAdapter:
    """
    Adapter for Onero Robot Arm (7-DOF) to be used with LeRobot.
    """

    config_class = OneroRobotConfig
    name = "onero_arm"

    def __init__(self, config: OneroRobotConfig):
        """
        Initialize the OneroAdapter with the given configuration.

        Args:
            config (OneroRobotConfig): Configuration object containing device path and other settings.
        """
        from lerobot.robots.robot import Robot

        Robot.__init__(self, config)
        self.config = config
        self._is_connected = False
        self._arm = None
        self._prev_observation = None
        self._gripper = None
        self._last_target_pos: list[float] | None = None
        self._last_send_time: float | None = None
        # Fallback cache for gripper status when the shared CAN bus returns a
        # malformed frame (C++ side raises ValueError: stoi). Initialized to
        # (open, no-force) which matches the training distribution.
        self._last_gripper_obs: tuple[float, float] = (100.0, 0.0)

        # Internal state for observation calculation
        self.dof = 7  # As per documentation

        # Initialize cameras from config
        if not config.is_teleop_leader:
            self.cameras = make_cameras_from_configs(config.cameras)

    def __str__(self) -> str:
        return f"onero {self.config.id}"

    def connect(self) -> None:
        """
        Connect to the Onero Arm robot.

        Establish communication with the robot arm hardware using the configuration provided.
        Enables motors upon successful connection. For follower arms, optionally moves to an
        intermediate position before the final home position to avoid collisions.

        Raises:
            DeviceAlreadyConnectedError: If the device is already connected.
            ConnectionError: If enabling motors fails.
            Exception: Propagates any other exceptions during connection.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. Setup API Config
        api_config = oneroarm.OneroConfig()
        api_config.device = self.config.port
        api_config.baud_rate = self.config.baud_rate

        api_config.robot_model = self.config.robot_model
        api_config.version = self.config.version
        api_config.model_description_path = self.config.onero_description_path
        api_config.slcan_type = self.config.slcan_type
        # Set other defaults or from config if needed
        api_config.dof = self.dof
        api_config.is_teleop_leader = self.config.is_teleop_leader
        if self.config.mit_kp is not None:
            api_config.mit_kp = self.config.mit_kp
        if self.config.mit_kd is not None:
            api_config.mit_kd = self.config.mit_kd

        try:
            # 2. Initialize and Connect Arm
            self._arm = oneroarm.OneroArm(api_config)

            # 3. Enable Motors
            if not self._arm.enable_motors():
                raise ConnectionError("Failed to enable motors")

            if self.config.enable_gripper:
                # 4. Optionally Initialize Gripper
                self._gripper = onerogripper.GripperControl()
                if not self._gripper.initialize(self.config.port, self.config.slcan_type):
                    logger.warning("Failed to initialize gripper, continuing without it.")
                else:
                    # Match Teleop2 init sequence: clear buffer -> enable motors.
                    # Without enable_motors(), set_position() silently does nothing.
                    try:
                        if hasattr(self._gripper, "clear_buffer_before_enable"):
                            self._gripper.clear_buffer_before_enable()
                            time.sleep(0.2)
                        self._gripper.enable_motors()
                        time.sleep(0.3)
                        # Seed the training-time initial gripper target (100 = closed in
                        # this setup) immediately after enable, otherwise the firmware
                        # holds whatever default it lands on and can drift during the
                        # arm's runtohome motion.
                        try:
                            self._gripper.set_position(100)
                            time.sleep(0.3)
                            logger.info(
                                f"{self} gripper seeded target=100 after enable; "
                                f"status={self._gripper.get_gripper_status().position:.1f}"
                            )
                        except Exception:
                            logger.exception("Failed to seed gripper target after enable")
                        logger.info(f"{self} gripper initialized and motors enabled.")
                    except Exception:
                        logger.exception("Failed to enable gripper motors")

            # Connect cameras
            if not self.config.is_teleop_leader:
                for cam_name, cam in self.cameras.items():
                    cam.connect()
                    logger.info(f"{self} camera '{cam_name}' connected.")

            self.is_connected = True
            logger.info(f"{self} connected.")

        except Exception as e:
            logger.error(f"Failed to connect to {self}: {e}")
            self.disconnect()
            raise e

    def disconnect(self) -> None:
        """
        Disconnect from the Onero Arm robot.

        Disables motors and tears down the connection to the hardware.
        """
        if not self.is_connected:
            return

        # Disconnect cameras
        if not self.config.is_teleop_leader:
            for cam_name, cam in self.cameras.items():
                if cam.is_connected:
                    cam.disconnect()
                    logger.info(f"{self} camera '{cam_name}' disconnected.")
        if self._arm:
            self._arm.disable_motors()
            self._arm = None

        self.is_connected = False
        logger.info(f"{self} disconnected.")

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is currently connected (arm + all cameras).

        Returns:
            bool: True if connected, False otherwise.
        """
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = copy.copy(action)

        # extract common parameters
        params = {
            "speed_scale": action.get("speed_scale", 1.0),
            "trajectory_connect": int(action.get("trajectory_connect", 0)),
            "block": action.get("block", True),
        }

        # define action handlers
        handlers: list[tuple[Callable, Callable]] = [
            (self._is_joint_control, self._execute_joint_control),
            (self._is_pose_control, self._execute_pose_control),
            (self._is_reset_control, self._execute_reset_position),
            (self._is_gripper_control, self._execute_gripper_control),
        ]

        # dispatch action
        for check_func, exec_func in handlers:
            if check_func(action):
                exec_func(action, params)
                return action

        return action

    # --- Predicate Functions ---

    def _is_joint_control(self, action: dict) -> bool:
        # Check if all joint data is present
        return all(f"joint{i}.pos" in action for i in range(1, self.dof + 1))

    def _is_pose_control(self, action: dict) -> bool:
        return "pose" in action

    def _is_reset_control(self, action: dict) -> bool:
        return action.get("reset", False)

    def _is_gripper_control(self, action: dict) -> bool:
        if not self.config.enable_gripper:
            return False
        return "gripper.position" in action

    # -- Execution Functions ---

    def _execute_joint_control(self, action: dict, params: dict):
        target_joints = [float(action[f"joint{i}.pos"]) for i in range(1, self.dof + 1)]
        target_vel = [0.0 for _ in range(self.dof)]

        _tgt_str = "[" + ", ".join(f"{v:+.4f}" for v in target_joints) + "]"
        _mj_t0 = time.perf_counter()
        self._arm.send_trajectory_point(target_joints, target_vel)
        self._arm.execute_buffered_trajectory()
        _mj_dt = (time.perf_counter() - _mj_t0) * 1000.0

        try:
            reached = self._arm.get_joint_positions_from_motors()
            _reached_str = "[" + ", ".join(f"{v:+.4f}" for v in reached) + "]"
            _resid_str = "[" + ", ".join(f"{reached[i] - target_joints[i]:+.4f}" for i in range(self.dof)) + "]"
        except Exception as e:
            _reached_str = f"<read-back failed: {e}>"
            _resid_str = "-"
        logger.info(
            f"[send->traj_pt] target={_tgt_str} send_dt={_mj_dt:.1f}ms "
            f"reached={_reached_str} resid(reached-target)={_resid_str}"
        )

        if self.config.enable_gripper and "gripper.position" in action:
            self._execute_gripper_control(action, params)

    def _execute_pose_control(self, action: dict, params: dict):
        target_pose = self._parse_pose(action["pose"])
        if target_pose:
            self._arm.movep(target_pose, **params)

    def _execute_reset_position(self):
        home_joints = list(self.config.home_joints_positions)
        self._arm.movej(home_joints, speed_scale=0.8, trajectory_connect=0)

    def _execute_gripper_control(self, action: dict, params: dict):
        if not self._gripper:
            logger.warning("Gripper control requested but gripper is not initialized.")
            return

        position = int(action["gripper.position"])
        try:
            self._gripper.set_position(position)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"gripper set_position({position}) failed ({e}); skipping this frame")

    # --- Utility Functions: Complex Data Conversion ---

    def _parse_pose(self, raw_pose) -> Any:
        """Convert various input formats to a oneroarm.Pose object"""
        if isinstance(raw_pose, oneroarm.Pose):
            return raw_pose

        target_pose = oneroarm.Pose()
        pose_arr = np.array(raw_pose)

        if pose_arr.size == 7:
            # [x, y, z, qw, qx, qy, qz]
            flat = pose_arr.flatten()
            target_pose.x, target_pose.y, target_pose.z = map(float, flat[:3])
            target_pose.qw, target_pose.qx, target_pose.qy, target_pose.qz = map(float, flat[3:])
        elif pose_arr.shape == (4, 4):
            # 4x4 Matrix
            translation = pose_arr[:3, 3]
            quat = t3d.quaternions.mat2quat(pose_arr[:3, :3])
            target_pose.x, target_pose.y, target_pose.z = map(float, translation)
            target_pose.qw, target_pose.qx, target_pose.qy, target_pose.qz = map(float, quat)
        else:
            # logger.warning(...)
            return None
        return target_pose

    def get_observation(self) -> dict[str, Any]:
        """
        Get the current observation of the robot state.

        Retrieves joint positions from the motors and computes joint velocities.

        Returns:
            Dict[str, Any]: Dictionary containing joint positions and velocities.
            Keys are 'joint1.pos', 'joint1.vel', etc.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        _arm_t0 = time.perf_counter()
        current_joints = self._arm.get_joint_positions_from_motors()
        current_joints_vel = self._arm.get_joint_velocities() if self.config.use_velocity else None
        _arm_dt = time.perf_counter() - _arm_t0

        for i in range(self.dof):
            # API uses 0-based index likely, joint names usually 1-based
            obs_dict[f"joint{i + 1}.pos"] = current_joints[i]
            if self.config.use_velocity:
                obs_dict[f"joint{i + 1}.vel"] = current_joints_vel[i]

        if self.config.enable_gripper:
            try:
                gripper_status = self._gripper.get_gripper_status()
                self._last_gripper_obs = (
                    float(gripper_status.position),
                    float(gripper_status.force),
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"gripper status read failed ({e}); using last cached value")
            pos, force = self._last_gripper_obs
            obs_dict["gripper.position"] = pos
            obs_dict["gripper.force"] = force

        # Capture images from cameras.
        # Instead of cam.async_read() which waits for a brand-new frame (bounded
        # by camera hw fps), we directly copy the latest cached frame that the
        # camera's background thread has already captured. This matches the
        # "take most recent" semantics used during recording and prevents the
        # control loop from being paced by camera fps.
        _cam_times = {}
        if not self.config.is_teleop_leader:
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                if cam.latest_frame is None:
                    # First frame not captured yet: block briefly to get one.
                    obs_dict[cam_key] = cam.async_read()
                else:
                    with cam.frame_lock:
                        obs_dict[cam_key] = cam.latest_frame.copy()
                dt_ms = (time.perf_counter() - start) * 1e3
                _cam_times[cam_key] = dt_ms
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        self._prev_observation = obs_dict

        _joints_str = "[" + ", ".join(f"{v:+.4f}" for v in current_joints) + "]"
        logger.info(
            f"[obs-timing] arm={_arm_dt*1000:.1f}ms "
            + " ".join(f"{k}={v:.1f}ms" for k, v in _cam_times.items())
            + f" joints={_joints_str}"
        )

        return obs_dict

    @property
    def _motors_ft(self) -> dict[str, type]:
        """
        Internal property defining the feature types for motors.
        """
        motors = {f"joint{i}.pos": float for i in range(1, self.dof + 1)}
        if self.config.enable_gripper:
            motors["gripper.position"] = float
        return motors

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """
        Internal property defining the feature types for cameras.
        Returns a dict mapping camera name to (height, width, channels) tuple.
        """
        if self.config.is_teleop_leader:
            return {}

        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def action_features(self) -> dict[str, type]:
        """
        Get the action space features.

        Returns:
            Dict[str, type]: Dictionary mapping action keys to their types.
        """
        return self._motors_ft

    @property
    def observation_features(self) -> dict[str, Any]:
        # depack observation features, avoid original dict mutation
        features = {**self._motors_ft}

        if self.config.use_velocity:
            for i in range(1, self.dof + 1):
                features[f"joint{i}.vel"] = float

        # Add camera features
        features.update(self._cameras_ft)

        return features

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True


class OneroTeleopFollower(OneroAdapter):
    """
    Specialized Adapter for Teleoperation scenarios.

    This adapter overrides send_action to prioritize direct MIT control.
    """

    def connect(self):
        super().connect()
        self.runtohome()
        time.sleep(5)  # Allow time for the robot to stabilize at home position after initialization
        return

    def unpack_action(self, action_tensor: torch.Tensor, dof: int = 7) -> dict:
        """
        unpack data from tensor to action dict for teleoperation.

        Args:
            action_tensor (torch.Tensor): A 1D tensor of shape (2 * dof,),
                                        where the first dof elements are positions and the last dof elements are velocities.
            dof (int): Degrees of freedom, default is 7.

        Returns:
            dict: A dictionary containing 'jointX.pos' and 'jointX.vel' keys.
        """
        action_list = action_tensor.tolist()

        # Sanity check
        if len(action_list) != dof * 2:
            logger.warning(f"Action tensor length {len(action_list)} does not match expected {dof * 2}")

        pos = action_list[:dof]
        vel = action_list[dof:]

        action_dict = {}
        for i in range(dof):
            joint_name = f"joint{i + 1}"
            action_dict[f"{joint_name}.pos"] = pos[i]
            action_dict[f"{joint_name}.vel"] = vel[i]

        return action_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to the robot with teleoperation-specific optimizations.

        Optimizations:
        1. Enforces non-blocking execution (block=False).
        2. Streamlines the control path (skips complex checks if simple joint control is detected).
        3. Robust error handling for high-frequency loops.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if action.get("reset", False):
            # Follower directly handles reset with fixed params
            self._execute_reset_position()
            return action

        # Teleop primarily relies on Joint Position Control and Velocity Control
        # Check if we have a standard joint position command and velocity command
        # (This check is slightly faster than calling the parent's generic handler loop)
        try:
            if action.get("mirror", False):
                action = self.mirror_action(action)
            r = range(1, self.dof + 1)
            target_pos = [float(action[f"joint{i}.pos"]) for i in r]
            target_vel = [0.0 for _ in r]

            # Push the setpoint, then immediately flush the trajectory buffer.
            # Teleop2's 100Hz loop appears to rely on an external MIT control
            # thread (the leader's ArmGravityCompensation loop) to consume
            # buffered points; in eval we have no such loop, so we trigger
            # execution ourselves after each send.
            self._arm.send_trajectory_point(target_pos, target_vel)
            self._arm.execute_buffered_trajectory()

            gripper_pos = action.get("gripper.position")
            if self.config.enable_gripper and gripper_pos is not None:
                try:
                    self._gripper.set_position(gripper_pos)
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"gripper set_position({gripper_pos}) failed ({e}); skipping")

        except KeyError:
            # Fall back to standard handler if not all joints are specified
            logger.error("Incomplete teleop action command, falling back to standard handler.")
            return super().send_action(action)

        except (ValueError, TypeError) as e:
            # Catch invalid value types
            logger.warning(f"Invalid joint values in action: {e}. Command ignored.")
            return action

        return action

    def mirror_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Mirror the action for teleoperation follower.
        """
        mirrored_action = copy.deepcopy(action)

        # Apply negation to joint positions and velocities
        for i in range(1, self.dof + 1):
            joint_pos = f"joint{i}.pos"
            if joint_pos in mirrored_action:
                mirrored_action[joint_pos] = -mirrored_action[joint_pos]
            joint_vel = f"joint{i}.vel"
            if joint_vel in mirrored_action:
                mirrored_action[joint_vel] = -mirrored_action[joint_vel]

        return mirrored_action

    """
    The runtohome method is designed to safely move the robot arm to its home position during initialization.
    For teleoperation followers, it's crucial to avoid any sudden or unsafe movements that could occur if
    """

    def runtohome(self):
        """
        Move the robot arm to the home position safely.
        """
        # 1. move to intermediate position if specified to avoid collisions

        intermediate_joints = getattr(self.config, "intermediate_joints_positions", None)
        if intermediate_joints is not None:
            try:
                logger.info("start movej to home")
                self._arm.movej(intermediate_joints, speed_scale=0.8, trajectory_connect=0)
            except Exception:
                logger.exception("Failed to move to intermediate joints position")

        # 2. move to final home position
        home_joints = getattr(self.config, "home_joints_positions", None)
        try:
            self._arm.movej(home_joints, speed_scale=0.8, trajectory_connect=0)
        except Exception:
            logger.exception("Failed to move to home joints position")

        # 3. optionally traverse a ready-pose waypoint sequence
        self.move_through_waypoints(getattr(self.config, "ready_waypoints", None))

        # 4. reset gripper to the training-distribution initial target (100 = closed here)
        if self.config.enable_gripper and self._gripper is not None:
            try:
                self._gripper.set_position(100)
                time.sleep(1.0)
                try:
                    pos = self._gripper.get_gripper_status().position
                    logger.info(f"{self} gripper reset target=100 readback={pos:.1f}")
                except Exception:
                    pass
            except Exception:
                logger.exception("Failed to reset gripper target during runtohome")
        return

    def move_through_waypoints(
        self,
        waypoints: list[list[float]] | None,
        speed_scale: float = 0.6,
        dwell: float = 1.0,
    ) -> None:
        """
        Sequentially move the follower arm through a list of joint waypoints.

        Mirrors the ready-pose motion used during Teleop2 recording so inference
        can drive the arm into a consistent starting pose before rollouts.
        """
        if not waypoints:
            return

        logger.info(f"moving through {len(waypoints)} ready waypoint(s)")
        for idx, wp in enumerate(waypoints):
            target = [float(x) for x in wp]
            try:
                self._arm.movej(target, speed_scale=speed_scale, trajectory_connect=0)
            except Exception:
                logger.exception(f"Failed to move to waypoint {idx}: {target}")
                return
            time.sleep(dwell)
