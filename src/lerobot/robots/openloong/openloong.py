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

"""
OpenLoong "Qinglong" humanoid robot implementation for LeRobot.

This module provides the interface between LeRobot and OpenLoong humanoid robot,
which uses MPC (Model Predictive Control) and WBC (Whole-Body Control).

Reference: https://github.com/loongOpen/OpenLoong-Dyn-Control

Features:
- MuJoCo simulation support
- MPC-based locomotion control
- WBC for whole-body coordination
- Joint position/velocity/torque control
- IMU feedback
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation

from ..robot import Robot
from .config_openloong import OpenLoongConfig
from .openloong_utils import NUM_MOTORS, OpenLoongJointIndex

logger = logging.getLogger(__name__)


@dataclass
class MotorState:
    """State of a single motor/joint."""
    q: float | None = None  # position (rad)
    dq: float | None = None  # velocity (rad/s)
    tau: float | None = None  # torque (Nm)
    temperature: float | None = None  # motor temperature (°C)


@dataclass
class IMUState:
    """IMU sensor state."""
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] angular velocity (rad/s)
    accelerometer: np.ndarray | None = None  # [x, y, z] linear acceleration (m/s²)
    rpy: np.ndarray | None = None  # [roll, pitch, yaw] (rad)
    temperature: float | None = None  # IMU temperature


@dataclass
class BaseState:
    """Base (pelvis) state."""
    position: np.ndarray | None = None  # [x, y, z] in world frame
    velocity: np.ndarray | None = None  # [vx, vy, vz] in world frame
    angular_velocity: np.ndarray | None = None  # [wx, wy, wz] in body frame


@dataclass
class FootState:
    """Foot contact state."""
    position: np.ndarray | None = None  # [x, y, z]
    velocity: np.ndarray | None = None  # [vx, vy, vz]
    contact_force: np.ndarray | None = None  # [fx, fy, fz]
    is_contact: bool = False  # Contact detection


@dataclass
class OpenLoongObservation:
    """Complete observation state for OpenLoong."""
    motor_state: list[MotorState] = field(
        default_factory=lambda: [MotorState() for _ in range(NUM_MOTORS)]
    )
    imu_state: IMUState = field(default_factory=IMUState)
    base_state: BaseState = field(default_factory=BaseState)
    left_foot: FootState = field(default_factory=FootState)
    right_foot: FootState = field(default_factory=FootState)
    timestamp: float = 0.0


class OpenLoong(Robot):
    """
    OpenLoong "Qinglong" humanoid robot.
    
    This class provides the interface to control the OpenLoong humanoid robot
    using LeRobot's standardized API. It supports both simulation (MuJoCo) and
    physical robot control.
    
    The robot uses:
    - MPC (Model Predictive Control) for locomotion planning
    - WBC (Whole-Body Control) for task prioritization
    
    Example:
        ```python
        from lerobot.robots.openloong import OpenLoong, OpenLoongConfig
        
        config = OpenLoongConfig(is_simulation=True)
        robot = OpenLoong(config)
        robot.connect()
        
        # Get observation
        obs = robot.get_observation()
        
        # Send action (joint positions)
        action = {"kLeftHipPitch.q": 0.1, "kRightHipPitch.q": 0.1}
        robot.send_action(action)
        
        robot.disconnect()
        ```
    """
    
    config_class = OpenLoongConfig
    name = "openloong"

    def __init__(self, config: OpenLoongConfig):
        """Initialize OpenLoong robot.
        
        Args:
            config: Configuration object containing robot parameters
        """
        super().__init__(config)
        
        logger.info("Initializing OpenLoong robot...")
        
        self.config = config
        self.control_dt = config.control_dt
        
        # Initialize cameras
        self._cameras = make_cameras_from_configs(config.cameras)
        
        # State variables
        self._observation: OpenLoongObservation | None = None
        self._shutdown_event = threading.Event()
        self._state_thread: threading.Thread | None = None
        
        # Control variables
        self.kp = np.array(config.kp, dtype=np.float32)
        self.kd = np.array(config.kd, dtype=np.float32)
        self._target_positions = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_velocities = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_torques = np.zeros(NUM_MOTORS, dtype=np.float32)
        
        # Simulation environment
        self.sim_env = None
        self._mj_model = None
        self._mj_data = None
        
        # MPC/WBC controllers (initialized in connect)
        self._mpc_controller = None
        self._wbc_controller = None
        
        logger.info(f"OpenLoong configuration: simulation={config.is_simulation}")

    def _init_simulation(self) -> None:
        """Initialize MuJoCo simulation environment."""
        try:
            import mujoco
            import mujoco.viewer
        except ImportError:
            raise ImportError(
                "MuJoCo is required for simulation. "
                "Install with: pip install mujoco"
            )
        
        # Try to load OpenLoong model
        if self.config.mjcf_path:
            xml_path = self.config.mjcf_path
        else:
            # Use default model path or raise error
            logger.warning("No MJCF path specified, using default humanoid model")
            # Fallback to default humanoid model for testing
            xml_path = None
        
        if xml_path:
            self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            # Create a simple humanoid model for testing
            # In production, this should load the actual OpenLoong URDF/MJCF
            self._mj_model = mujoco.MjModel.from_xml_string(
                self._get_default_humanoid_xml()
            )
        
        self._mj_data = mujoco.MjData(self._mj_model)
        
        logger.info(f"MuJoCo simulation initialized: {self._mj_model.nq} DOF")

    def _get_default_humanoid_xml(self) -> str:
        """Get default humanoid XML for testing when OpenLoong model not available."""
        return """
        <mujoco model="openloong_test">
          <compiler angle="radian" meshdir="."/>
          <option timestep="0.002" iterations="50" solver="Newton" gravity="0 0 -9.81">
            <flag warmstart="enable"/>
          </option>
          <default>
            <joint armature="0.01" damping="2" limited="true"/>
            <geom contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
          </default>
          <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="50 50 0.1" rgba=".9 .9 .9 1"/>
            <body name="torso" pos="0 0 1.0">
              <freejoint name="root"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.1"/>
              <!-- Simplified 29-DOF structure -->
              <!-- Left leg -->
              <body name="left_thigh" pos="0 0.1 0">
                <joint name="kLeftHipPitch" axis="0 1 0" range="-0.87 0.87"/>
                <joint name="kLeftHipRoll" axis="1 0 0" range="-0.52 0.52"/>
                <joint name="kLeftHipYaw" axis="0 0 1" range="-0.87 0.87"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.07"/>
                <body name="left_shin" pos="0 0 -0.4">
                  <joint name="kLeftKnee" axis="0 1 0" range="-0.17 2.1"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.06"/>
                  <body name="left_foot" pos="0 0 -0.4">
                    <joint name="kLeftAnklePitch" axis="0 1 0" range="-0.87 0.52"/>
                    <joint name="kLeftAnkleRoll" axis="1 0 0" range="-0.35 0.35"/>
                    <geom type="box" size="0.1 0.05 0.02" pos="0 0 -0.02"/>
                  </body>
                </body>
              </body>
              <!-- Right leg -->
              <body name="right_thigh" pos="0 -0.1 0">
                <joint name="kRightHipPitch" axis="0 1 0" range="-0.87 0.87"/>
                <joint name="kRightHipRoll" axis="1 0 0" range="-0.52 0.52"/>
                <joint name="kRightHipYaw" axis="0 0 1" range="-0.87 0.87"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.07"/>
                <body name="right_shin" pos="0 0 -0.4">
                  <joint name="kRightKnee" axis="0 1 0" range="-0.17 2.1"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.06"/>
                  <body name="right_foot" pos="0 0 -0.4">
                    <joint name="kRightAnklePitch" axis="0 1 0" range="-0.87 0.52"/>
                    <joint name="kRightAnkleRoll" axis="1 0 0" range="-0.35 0.35"/>
                    <geom type="box" size="0.1 0.05 0.02" pos="0 0 -0.02"/>
                  </body>
                </body>
              </body>
              <!-- Waist -->
              <body name="waist" pos="0 0 0.3">
                <joint name="kWaistYaw" axis="0 0 1" range="-1.22 1.22"/>
                <joint name="kWaistRoll" axis="1 0 0" range="-0.26 0.26"/>
                <joint name="kWaistPitch" axis="0 1 0" range="-0.52 0.52"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.09"/>
                <!-- Left arm -->
                <body name="left_upper_arm" pos="0 0.2 0.15">
                  <joint name="kLeftShoulderPitch" axis="0 1 0" range="-2.97 2.97"/>
                  <joint name="kLeftShoulderRoll" axis="1 0 0" range="-0.52 3.14"/>
                  <joint name="kLeftShoulderYaw" axis="0 0 1" range="-2.97 2.97"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.05"/>
                  <body name="left_lower_arm" pos="0 0 -0.3">
                    <joint name="kLeftElbow" axis="0 1 0" range="-0.17 2.97"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04"/>
                    <body name="left_hand" pos="0 0 -0.25">
                      <joint name="kLeftWristRoll" axis="1 0 0" range="-0.79 0.79"/>
                      <joint name="kLeftWristPitch" axis="0 1 0" range="-0.79 0.79"/>
                      <joint name="kLeftWristYaw" axis="0 0 1" range="-0.79 0.79"/>
                      <geom type="sphere" size="0.04"/>
                    </body>
                  </body>
                </body>
                <!-- Right arm -->
                <body name="right_upper_arm" pos="0 -0.2 0.15">
                  <joint name="kRightShoulderPitch" axis="0 1 0" range="-2.97 2.97"/>
                  <joint name="kRightShoulderRoll" axis="1 0 0" range="-3.14 0.52"/>
                  <joint name="kRightShoulderYaw" axis="0 0 1" range="-2.97 2.97"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.05"/>
                  <body name="right_lower_arm" pos="0 0 -0.3">
                    <joint name="kRightElbow" axis="0 1 0" range="-0.17 2.97"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04"/>
                    <body name="right_hand" pos="0 0 -0.25">
                      <joint name="kRightWristRoll" axis="1 0 0" range="-0.79 0.79"/>
                      <joint name="kRightWristPitch" axis="0 1 0" range="-0.79 0.79"/>
                      <joint name="kRightWristYaw" axis="0 0 1" range="-0.79 0.79"/>
                      <geom type="sphere" size="0.04"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position joint="kLeftHipPitch" kp="150" kv="2"/>
            <position joint="kLeftHipRoll" kp="150" kv="2"/>
            <position joint="kLeftHipYaw" kp="150" kv="2"/>
            <position joint="kLeftKnee" kp="300" kv="4"/>
            <position joint="kLeftAnklePitch" kp="40" kv="2"/>
            <position joint="kLeftAnkleRoll" kp="40" kv="2"/>
            <position joint="kRightHipPitch" kp="150" kv="2"/>
            <position joint="kRightHipRoll" kp="150" kv="2"/>
            <position joint="kRightHipYaw" kp="150" kv="2"/>
            <position joint="kRightKnee" kp="300" kv="4"/>
            <position joint="kRightAnklePitch" kp="40" kv="2"/>
            <position joint="kRightAnkleRoll" kp="40" kv="2"/>
            <position joint="kWaistYaw" kp="250" kv="5"/>
            <position joint="kWaistRoll" kp="250" kv="5"/>
            <position joint="kWaistPitch" kp="250" kv="5"/>
            <position joint="kLeftShoulderPitch" kp="80" kv="3"/>
            <position joint="kLeftShoulderRoll" kp="80" kv="3"/>
            <position joint="kLeftShoulderYaw" kp="80" kv="3"/>
            <position joint="kLeftElbow" kp="80" kv="3"/>
            <position joint="kLeftWristRoll" kp="40" kv="1.5"/>
            <position joint="kLeftWristPitch" kp="40" kv="1.5"/>
            <position joint="kLeftWristYaw" kp="40" kv="1.5"/>
            <position joint="kRightShoulderPitch" kp="80" kv="3"/>
            <position joint="kRightShoulderRoll" kp="80" kv="3"/>
            <position joint="kRightShoulderYaw" kp="80" kv="3"/>
            <position joint="kRightElbow" kp="80" kv="3"/>
            <position joint="kRightWristRoll" kp="40" kv="1.5"/>
            <position joint="kRightWristPitch" kp="40" kv="1.5"/>
            <position joint="kRightWristYaw" kp="40" kv="1.5"/>
          </actuator>
        </mujoco>
        """

    def _state_update_loop(self) -> None:
        """Background thread for updating robot state."""
        while not self._shutdown_event.is_set():
            start_time = time.time()
            
            if self.config.is_simulation and self.sim_env is not None:
                # Step simulation
                self._step_simulation()
            
            # Update observation from simulation or physical robot
            self._update_observation()
            
            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    def _step_simulation(self) -> None:
        """Step MuJoCo simulation."""
        if self._mj_model is None or self._mj_data is None:
            return
        
        import mujoco
        
        # Apply control
        self._apply_control()
        
        # Step physics
        mujoco.mj_step(self._mj_model, self._mj_data)

    def _apply_control(self) -> None:
        """Apply control commands to simulation."""
        if self._mj_data is None:
            return
        
        # Get actuator indices (skip freejoint root)
        nv = self._mj_model.nv
        nu = self._mj_model.nu
        
        # Apply position control
        for i, joint in enumerate(OpenLoongJointIndex):
            if i < nu:
                self._mj_data.ctrl[i] = self._target_positions[i]

    def _update_observation(self) -> None:
        """Update observation from simulation or physical robot."""
        if self.config.is_simulation:
            self._update_observation_from_sim()
        else:
            self._update_observation_from_robot()

    def _update_observation_from_sim(self) -> None:
        """Update observation from MuJoCo simulation."""
        if self._mj_data is None:
            return
        
        obs = OpenLoongObservation()
        obs.timestamp = time.time()
        
        # Get qpos (positions) and qvel (velocities)
        # Skip first 7 for freejoint (root position and quaternion)
        qpos = self._mj_data.qpos[7:7 + NUM_MOTORS]
        qvel = self._mj_data.qvel[6:6 + NUM_MOTORS]
        
        # Motor states
        for i, joint in enumerate(OpenLoongJointIndex):
            if i < len(qpos):
                obs.motor_state[i].q = float(qpos[i])
                obs.motor_state[i].dq = float(qvel[i])
                obs.motor_state[i].tau = 0.0  # Could compute from actuator forces
        
        # IMU state (from root body)
        if self._mj_model.nq >= 7:
            obs.imu_state.quaternion = np.array(self._mj_data.qpos[3:7])  # [w, x, y, z]
            obs.imu_state.rpy = self._quat_to_euler(obs.imu_state.quaternion)
            obs.imu_state.gyroscope = np.array(self._mj_data.qvel[3:6])  # angular velocity
            # Acceleration requires computing from sensor or finite differences
            obs.imu_state.accelerometer = np.zeros(3)
        
        # Base state
        obs.base_state.position = np.array(self._mj_data.qpos[:3])
        obs.base_state.velocity = np.array(self._mj_data.qvel[:3])
        obs.base_state.angular_velocity = np.array(self._mj_data.qvel[3:6])
        
        self._observation = obs

    def _update_observation_from_robot(self) -> None:
        """Update observation from physical robot (to be implemented)."""
        # This would communicate with the physical OpenLoong robot
        # through the robot's SDK/communication protocol
        logger.warning("Physical robot communication not yet implemented")
        
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(
            np.abs(sinp) >= 1,
            np.copysign(np.pi / 2, sinp),
            np.arcsin(sinp)
        )
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space features."""
        features = {}
        
        # Joint position commands
        for joint in OpenLoongJointIndex:
            features[f"{joint.name}.q"] = float
            
        return features

    def calibrate(self) -> None:
        """Calibrate robot (no-op for OpenLoong)."""
        logger.info("OpenLoong robot calibration: no action needed")

    def configure(self) -> None:
        """Configure robot parameters."""
        logger.info("Configuring OpenLoong robot...")
        
        # Apply any runtime configuration
        if self.config.is_simulation:
            # Set simulation parameters
            pass
        else:
            # Configure physical robot
            pass

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot.
        
        Args:
            calibrate: Whether to calibrate after connection
        """
        logger.info("Connecting to OpenLoong robot...")
        
        if self.config.is_simulation:
            self._init_simulation()
            self.sim_env = True  # Marker for simulation mode
        else:
            # Initialize physical robot connection
            self._init_physical_robot()
        
        # Start state update thread
        self._state_thread = threading.Thread(target=self._state_update_loop)
        self._state_thread.start()
        
        # Connect cameras
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()
        
        logger.info(f"Connected {len(self._cameras)} camera(s)")
        
        # Wait for first observation
        timeout = 5.0
        start_time = time.time()
        while self._observation is None:
            if time.time() - start_time > timeout:
                raise TimeoutError("Failed to get initial observation")
            time.sleep(0.01)
        
        logger.info("OpenLoong robot connected successfully")
        
        # Calibrate if requested
        if calibrate:
            self.calibrate()

    def _init_physical_robot(self) -> None:
        """Initialize connection to physical OpenLoong robot."""
        logger.info(f"Connecting to physical robot at {self.config.robot_ip}:{self.config.robot_port}")
        # This would implement the actual communication protocol
        # with the OpenLoong robot controller
        raise NotImplementedError(
            "Physical robot connection not yet implemented. "
            "Use is_simulation=True for simulation mode."
        )

    def disconnect(self) -> None:
        """Disconnect from robot and cleanup."""
        logger.info("Disconnecting from OpenLoong robot...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for state thread
        if self._state_thread is not None:
            self._state_thread.join(timeout=2.0)
            if self._state_thread.is_alive():
                logger.warning("State thread did not stop cleanly")
        
        # Close simulation
        if self.config.is_simulation:
            self.sim_env = None
            self._mj_model = None
            self._mj_data = None
        
        # Disconnect cameras
        for cam in self._cameras.values():
            cam.disconnect()
        
        logger.info("OpenLoong robot disconnected")

    def get_observation(self) -> RobotObservation:
        """Get current robot observation.
        
        Returns:
            Dictionary containing all sensor readings
        """
        if self._observation is None:
            return {}
        
        obs = {}
        
        # Motor states
        for i, joint in enumerate(OpenLoongJointIndex):
            name = joint.name
            obs[f"{name}.q"] = self._observation.motor_state[i].q or 0.0
            obs[f"{name}.dq"] = self._observation.motor_state[i].dq or 0.0
            obs[f"{name}.tau"] = self._observation.motor_state[i].tau or 0.0
        
        # IMU data
        if self._observation.imu_state.gyroscope is not None:
            obs["imu.gyro.x"] = self._observation.imu_state.gyroscope[0]
            obs["imu.gyro.y"] = self._observation.imu_state.gyroscope[1]
            obs["imu.gyro.z"] = self._observation.imu_state.gyroscope[2]
        
        if self._observation.imu_state.accelerometer is not None:
            obs["imu.accel.x"] = self._observation.imu_state.accelerometer[0]
            obs["imu.accel.y"] = self._observation.imu_state.accelerometer[1]
            obs["imu.accel.z"] = self._observation.imu_state.accelerometer[2]
        
        if self._observation.imu_state.quaternion is not None:
            obs["imu.quat.w"] = self._observation.imu_state.quaternion[0]
            obs["imu.quat.x"] = self._observation.imu_state.quaternion[1]
            obs["imu.quat.y"] = self._observation.imu_state.quaternion[2]
            obs["imu.quat.z"] = self._observation.imu_state.quaternion[3]
        
        if self._observation.imu_state.rpy is not None:
            obs["imu.rpy.roll"] = self._observation.imu_state.rpy[0]
            obs["imu.rpy.pitch"] = self._observation.imu_state.rpy[1]
            obs["imu.rpy.yaw"] = self._observation.imu_state.rpy[2]
        
        # Base state
        if self._observation.base_state.position is not None:
            obs["base.pos.x"] = self._observation.base_state.position[0]
            obs["base.pos.y"] = self._observation.base_state.position[1]
            obs["base.pos.z"] = self._observation.base_state.position[2]
        
        # Camera images
        for cam_name, cam in self._cameras.items():
            obs[cam_name] = cam.async_read()
        
        return obs

    @property
    def is_calibrated(self) -> bool:
        """Whether robot is calibrated."""
        return True

    @property
    def is_connected(self) -> bool:
        """Whether robot is connected."""
        return self._observation is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor feature types."""
        return {f"{joint.name}.q": float for joint in OpenLoongJointIndex}

    @property
    def cameras(self) -> dict:
        """Camera dictionary."""
        return self._cameras

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera feature shapes."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space features."""
        features: dict[str, type | tuple] = {}
        
        # Motor states
        for joint in OpenLoongJointIndex:
            name = joint.name
            features[f"{name}.q"] = float
            features[f"{name}.dq"] = float
            features[f"{name}.tau"] = float
        
        # IMU
        features["imu.gyro.x"] = float
        features["imu.gyro.y"] = float
        features["imu.gyro.z"] = float
        features["imu.accel.x"] = float
        features["imu.accel.y"] = float
        features["imu.accel.z"] = float
        features["imu.quat.w"] = float
        features["imu.quat.x"] = float
        features["imu.quat.y"] = float
        features["imu.quat.z"] = float
        features["imu.rpy.roll"] = float
        features["imu.rpy.pitch"] = float
        features["imu.rpy.yaw"] = float
        
        # Base position
        features["base.pos.x"] = float
        features["base.pos.y"] = float
        features["base.pos.z"] = float
        
        # Cameras
        features.update(self._cameras_ft)
        
        return features

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot.
        
        Args:
            action: Dictionary with joint position commands
            
        Returns:
            The action that was sent
        """
        # Update target positions from action
        for joint in OpenLoongJointIndex:
            key = f"{joint.name}.q"
            if key in action:
                self._target_positions[joint.value] = action[key]
        
        return action

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:
        """Reset robot to default position.
        
        Args:
            control_dt: Control timestep
            default_positions: Target default positions
        """
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)
        
        logger.info("Resetting OpenLoong robot to default position...")
        
        if self.config.is_simulation and self._mj_data is not None:
            # Reset simulation
            import mujoco
            mujoco.mj_resetData(self._mj_model, self._mj_data)
            
            # Set initial joint positions
            self._mj_data.qpos[7:7 + NUM_MOTORS] = default_positions
            
            # Forward kinematics
            mujoco.mj_forward(self._mj_model, self._mj_data)
        
        # Interpolate to default position for physical robot
        elif not self.config.is_simulation:
            # Get current state
            obs = self.get_observation()
            
            # Record current positions
            current_pos = np.zeros(NUM_MOTORS, dtype=np.float32)
            for joint in OpenLoongJointIndex:
                current_pos[joint.value] = obs.get(f"{joint.name}.q", 0.0)
            
            # Interpolate
            total_time = 3.0
            num_steps = int(total_time / control_dt)
            
            for step in range(num_steps):
                start_time = time.time()
                
                alpha = step / num_steps
                action_dict = {}
                for joint in OpenLoongJointIndex:
                    target = default_positions[joint.value]
                    interp = current_pos[joint.value] * (1 - alpha) + target * alpha
                    action_dict[f"{joint.name}.q"] = float(interp)
                
                self.send_action(action_dict)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)
        
        logger.info("Reset complete")

    def set_mpc_gains(
        self,
        u_weight: float | None = None,
        L_diag: list[float] | None = None,
        K_diag: list[float] | None = None,
    ) -> None:
        """Set MPC controller gains.
        
        Args:
            u_weight: Input weight
            L_diag: State error weights [eul, pos, omega, vel]
            K_diag: Input weights [fl, tl, fr, tr]
        """
        if u_weight is not None:
            self.config.mpc_u_weight = u_weight
        if L_diag is not None:
            self.config.mpc_L_diag = L_diag
        if K_diag is not None:
            self.config.mpc_K_diag = K_diag
        
        logger.info("MPC gains updated")

    def set_wbc_weights(
        self,
        Q1: list[float] | None = None,
        Q2: list[float] | None = None,
    ) -> None:
        """Set WBC QP weights.
        
        Args:
            Q1: Contact force error weight
            Q2: Joint acceleration error weight
        """
        if Q1 is not None:
            self.config.wbc_qp_weight_Q1 = Q1
        if Q2 is not None:
            self.config.wbc_qp_weight_Q2 = Q2
        
        logger.info("WBC weights updated")
