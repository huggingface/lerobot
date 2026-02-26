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
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.envs.factory import make_env
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
from lerobot.robots.unitree_g1.locomotion.gr00t_locomotion import GrootLocomotionController
from lerobot.robots.unitree_g1.locomotion.holosoma_locomotion import HolosomaLocomotionController
from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)



# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"


@dataclass
class MotorState:
    q: float | None = None  # position
    dq: float | None = None  # velocity
    tau_est: float | None = None  # estimated torque
    temperature: float | None = None  # motor temperature


@dataclass
class IMUState:
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] angular velocity (rad/s)
    accelerometer: np.ndarray | None = None  # [x, y, z] linear acceleration (m/s²)
    rpy: np.ndarray | None = None  # [roll, pitch, yaw] (rad)
    temperature: float | None = None  # IMU temperature


# g1 observation class
@dataclass
class G1_29_LowState:  # noqa: N801
    motor_state: list[MotorState] = field(default_factory=lambda: [MotorState() for _ in G1_29_JointIndex])
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: Any = None  # Raw wireless remote data
    mode_machine: int = 0  # Robot mode


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")
        logger.info(f"Config: is_simulation={config.is_simulation}, robot_ip={config.robot_ip}, locomotion='{config.locomotion}'")

        self.config = config
        self.control_dt = config.control_dt

        # Initialize cameras config (ZMQ-based) - actual connection in connect()
        self._cameras = make_cameras_from_configs(config.cameras)

        # Import channel classes based on mode
        if config.is_simulation:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
        else:
            from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )

        # Store for use in connect()
        self._ChannelFactoryInitialize = ChannelFactoryInitialize
        self._ChannelPublisher = ChannelPublisher
        self._ChannelSubscriber = ChannelSubscriber

        # Initialize state variables
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None

        # Only initialize IK if gravity compensation is enabled (requires casadi/pinocchio)
        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        # Locomotion controller (groot or holosoma)
        self.locomotion_controller = None
        if config.locomotion == "groot":
            self.locomotion_controller = GrootLocomotionController()
        elif config.locomotion == "holosoma":
            self.locomotion_controller = HolosomaLocomotionController()

        # Locomotion thread state
        self._locomotion_thread = None
        self._locomotion_action_lock = threading.Lock()
        self._latest_joystick_action = {
            "remote.lx": 0.0,
            "remote.ly": 0.0,
            "remote.rx": 0.0,
            "remote.ry": 0.0,
        }
        self._latest_locomotion_action = {}  # Stores last locomotion output

    def _subscribe_motor_state(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                try:
                    self.sim_env.step()
                except ValueError as e:
                    # Handle "zero norm quaternion" error during sim initialization
                    if "quaternion" in str(e).lower():
                        continue  # Skip this step, try again next iteration
                    raise

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # Capture motor states using jointindex
                for id in G1_29_JointIndex:
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = msg.motor_state[id].tau_est
                    lowstate.motor_state[id].temperature = msg.motor_state[id].temperature

                # Capture IMU state
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # Capture wireless remote data
                lowstate.wireless_remote = msg.wireless_remote

                # Capture mode_machine
                lowstate.mode_machine = msg.mode_machine

                self._lowstate = lowstate

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # maintain constant control dt
            time.sleep(sleep_time)

    def _locomotion_loop(self):
        """Background thread that runs locomotion at policy's control_dt."""
        control_dt = self.locomotion_controller.control_dt
        logger.info(f"Locomotion loop starting with control_dt={control_dt} ({1.0/control_dt:.1f}Hz)")

        loop_count = 0
        last_log_time = time.time()

        while not self._shutdown_event.is_set():
            start_time = time.time()

            if self._lowstate is not None and self.locomotion_controller is not None:
                loop_count += 1
                if time.time() - last_log_time >= 5.0:  # Log every 5 seconds
                    actual_hz = loop_count / (time.time() - last_log_time)
                    logger.info(f"Locomotion actual rate: {actual_hz:.1f}Hz (target: {1.0/control_dt:.1f}Hz)")
                    loop_count = 0
                    last_log_time = time.time()
                # Get latest joystick action
                with self._locomotion_action_lock:
                    action = self._latest_joystick_action.copy()

                # Run locomotion step
                locomotion_action = self.locomotion_controller.run_step(action, self._lowstate)

                # Store for send_action to retrieve
                with self._locomotion_action_lock:
                    self._latest_locomotion_action = locomotion_action.copy()

                # Send locomotion commands (legs + waist, indices 0-14)
                for motor in G1_29_JointIndex:
                    if motor.value > 14:  # Skip arm motors
                        continue
                    key = f"{motor.name}.q"
                    if key in locomotion_action:
                        self.msg.motor_cmd[motor.value].q = locomotion_action[key]
                        self.msg.motor_cmd[motor.value].qd = 0
                        self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                        self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                        self.msg.motor_cmd[motor.value].tau = 0

                self.msg.crc = self.crc.Crc(self.msg)
                self.lowcmd_publisher.Write(self.msg)

            elapsed = time.time() - start_time
            sleep_time = max(0, control_dt - elapsed)
            time.sleep(sleep_time)

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.locomotion_controller is None:
            return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

        arm_features = {f"{G1_29_JointArmIndex(motor).name}.q": float for motor in G1_29_JointArmIndex}
        remote_features = {
            "remote.lx": float,
            "remote.ly": float,
            "remote.rx": float,
            "remote.ry": float,
        }
        return {**arm_features, **remote_features}

    def calibrate(self) -> None:  # robot is already calibrated
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowCmd_ as hg_LowCmd,
            LowState_ as hg_LowState,
        )
        from unitree_sdk2py.utils.crc import CRC

        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            self._ChannelFactoryInitialize(0, "lo")
            self._env_wrapper = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            # Extract the actual gym env from the dict structure
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        else:
            self._ChannelFactoryInitialize(0, config=self.config)

        # Initialize direct motor control interface
        self.lowcmd_publisher = self._ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = self._ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        # Start subscribe thread to read robot state
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.start()

        # Connect cameras
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()

        logger.info(f"Connected {len(self._cameras)} camera(s).")

        # Initialize lowcmd message
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0

        # Wait for first state message to arrive
        lowstate = None
        while lowstate is None:
            lowstate = self._lowstate
            if lowstate is None:
                time.sleep(0.01)
            logger.warning("[UnitreeG1] Waiting for robot state...")
        logger.warning("[UnitreeG1] Connected to robot.")
        self.msg.mode_machine = lowstate.mode_machine

        # Initialize kp/kd from locomotion controller for legs/waist, config for arms
        if self.locomotion_controller is not None and hasattr(self.locomotion_controller, 'kp'):
            # Use locomotion controller gains for legs/waist (0-14), config gains for arms (15-28)
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)
            # Override legs and waist with locomotion controller gains
            self.kp[:15] = self.locomotion_controller.kp[:15]
            self.kd[:15] = self.locomotion_controller.kd[:15]
            logger.info(f"Using KP/KD from locomotion controller (legs/waist) + config (arms)")
            logger.info(f"  Legs KP: {self.kp[:12].tolist()}")
            logger.info(f"  Arms KP: {self.kp[15:].tolist()}")
        else:
            # Use default from config
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)
            logger.info(f"Using KP/KD from config")

        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp[id.value]
            self.msg.motor_cmd[id].kd = self.kd[id.value]
            self.msg.motor_cmd[id].q = lowstate.motor_state[id.value].q

        # Start locomotion thread if enabled
        if self.locomotion_controller is not None:
            self._locomotion_thread = threading.Thread(target=self._locomotion_loop, daemon=True)
            self._locomotion_thread.start()
            fps = int(1.0 / self.locomotion_controller.control_dt)
            logger.info(f"Locomotion thread started ({fps}Hz)")

    def disconnect(self):
        # Signal thread to stop and unblock any waits
        self._shutdown_event.set()

        # Wait for subscribe thread to finish
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not stop cleanly")

        # Wait for locomotion thread to finish
        if self._locomotion_thread is not None:
            self._locomotion_thread.join(timeout=2.0)
            if self._locomotion_thread.is_alive():
                logger.warning("Locomotion thread did not stop cleanly")

        # Close simulation environment
        if self.config.is_simulation and self.sim_env is not None:
            try:
                # Force-kill the image publish subprocess first to avoid long waits
                if hasattr(self.sim_env, "simulator") and hasattr(self.sim_env.simulator, "sim_env"):
                    sim_env_inner = self.sim_env.simulator.sim_env
                    if hasattr(sim_env_inner, "image_publish_process"):
                        proc = sim_env_inner.image_publish_process
                        if proc.process and proc.process.is_alive():
                            logger.info("Force-terminating image publish subprocess...")
                            proc.stop_event.set()
                            proc.process.terminate()
                            proc.process.join(timeout=1)
                            if proc.process.is_alive():
                                proc.process.kill()
                self.sim_env.close()
            except Exception as e:
                logger.warning(f"Error closing sim_env: {e}")
            self.sim_env = None
            self._env_wrapper = None

        # Disconnect cameras
        for cam in self._cameras.values():
            cam.disconnect()

    def get_observation(self) -> RobotObservation:
        lowstate = self._lowstate
        if lowstate is None:
            return {}

        obs = {}

        # Motors - q, dq, tau for all joints
        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            obs[f"{name}.q"] = lowstate.motor_state[idx].q
            obs[f"{name}.dq"] = lowstate.motor_state[idx].dq
            obs[f"{name}.tau"] = lowstate.motor_state[idx].tau_est

        # IMU - gyroscope
        if lowstate.imu_state.gyroscope:
            obs["imu.gyro.x"] = lowstate.imu_state.gyroscope[0]
            obs["imu.gyro.y"] = lowstate.imu_state.gyroscope[1]
            obs["imu.gyro.z"] = lowstate.imu_state.gyroscope[2]

        # IMU - accelerometer
        if lowstate.imu_state.accelerometer:
            obs["imu.accel.x"] = lowstate.imu_state.accelerometer[0]
            obs["imu.accel.y"] = lowstate.imu_state.accelerometer[1]
            obs["imu.accel.z"] = lowstate.imu_state.accelerometer[2]

        # IMU - quaternion
        if lowstate.imu_state.quaternion:
            obs["imu.quat.w"] = lowstate.imu_state.quaternion[0]
            obs["imu.quat.x"] = lowstate.imu_state.quaternion[1]
            obs["imu.quat.y"] = lowstate.imu_state.quaternion[2]
            obs["imu.quat.z"] = lowstate.imu_state.quaternion[3]

        # IMU - rpy
        if lowstate.imu_state.rpy:
            obs["imu.rpy.roll"] = lowstate.imu_state.rpy[0]
            obs["imu.rpy.pitch"] = lowstate.imu_state.rpy[1]
            obs["imu.rpy.yaw"] = lowstate.imu_state.rpy[2]

        # Wireless remote (raw bytes for teleoperator)
        if lowstate.wireless_remote:
            obs["wireless_remote"] = lowstate.wireless_remote

        # Cameras - read images from ZMQ cameras
        for cam_name, cam in self._cameras.items():
            frame = cam.async_read()
            obs[cam_name] = frame

        return obs

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Joint positions for all 29 joints."""
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

    @property
    def _torques_ft(self) -> dict[str, type]:
        """Joint torques for arm joints only (14 joints, indices 15-28)."""
        return {f"{G1_29_JointArmIndex(motor).name}.tau": float for motor in G1_29_JointArmIndex}

    @property
    def cameras(self) -> dict:
        return self._cameras

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def policy_output_action_features(self, action_dim: int) -> dict[str, type]:
        """Return action feature names expected for a policy output vector."""
        if action_dim == 18:
            arm_joint_names = [f"{joint.name}.q" for joint in G1_29_JointArmIndex]
            remote_names = ["remote.lx", "remote.ly", "remote.rx", "remote.ry"]
            return {name: float for name in arm_joint_names + remote_names}

        if action_dim == len(self.action_features):
            return {name: float for name in self.action_features}

        return {f"action.{i}": float for i in range(action_dim)}

    def decode_policy_action(self, action_vector: Any) -> RobotAction:
        """Decode a policy output vector into the robot action dict format."""
        arr = action_vector
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        if hasattr(arr, "numpy"):
            arr = arr.numpy()

        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        action_features = self.policy_output_action_features(arr.shape[0])
        keys = list(action_features.keys())
        return {key: float(arr[i]) for i, key in enumerate(keys)}

    def _update_locomotion_action(self, action: RobotAction) -> None:
        """Update the joystick state for the locomotion thread."""
        with self._locomotion_action_lock:
            for key in ["remote.lx", "remote.ly", "remote.rx", "remote.ry"]:
                if key in action:
                    self._latest_joystick_action[key] = action[key]
            # Also copy button states if present
            for i in range(16):
                btn_key = f"remote.button.{i}"
                if btn_key in action:
                    self._latest_joystick_action[btn_key] = action[btn_key]

    def send_action(self, action: RobotAction | Any) -> RobotAction:
        if not isinstance(action, dict):
            action = self.decode_policy_action(action)
        elif "action.0" in action:
            generic_keys = sorted(
                (key for key in action.keys() if key.startswith("action.")),
                key=lambda key: int(key.split(".")[1]),
            )
            action = self.decode_policy_action([action[key] for key in generic_keys])

        # If locomotion is enabled, update joystick state and only send arm commands
        if self.locomotion_controller is not None:
            # Update joystick state for locomotion thread
            self._update_locomotion_action(action)

            # Get latest locomotion action and merge (locomotion first, then teleop overrides)
            with self._locomotion_action_lock:
                locomotion_action = self._latest_locomotion_action.copy()
            merged = dict(locomotion_action)
            for key, value in action.items():
                merged[key] = value
            action = merged

            # Only send arm motor commands (indices 15-28)
            # Locomotion thread handles legs/waist (indices 0-14)
            for motor in G1_29_JointIndex:
                if motor.value < 15:
                    continue
                key = f"{motor.name}.q"
                if key in action:
                    self.msg.motor_cmd[motor.value].q = action[key]
                    self.msg.motor_cmd[motor.value].qd = 0
                    self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                    self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                    self.msg.motor_cmd[motor.value].tau = 0

            if self.config.gravity_compensation:
                action_np = np.zeros(14)
                arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value
                for joint in G1_29_JointArmIndex:
                    local_idx = joint.value - arm_start_idx
                    action_np[local_idx] = self.msg.motor_cmd[joint.value].q
                tau = self.arm_ik.solve_tau(action_np)
                for joint in G1_29_JointArmIndex:
                    local_idx = joint.value - arm_start_idx
                    self.msg.motor_cmd[joint.value].tau = tau[local_idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
        else:
            # No locomotion - send all motor commands
            for motor in G1_29_JointIndex:
                key = f"{motor.name}.q"
                if key in action:
                    self.msg.motor_cmd[motor.value].q = action[key]
                    self.msg.motor_cmd[motor.value].qd = 0
                    self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                    self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                    self.msg.motor_cmd[motor.value].tau = 0

            if self.config.gravity_compensation:
                action_np = np.zeros(14)
                arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value
                for joint in G1_29_JointArmIndex:
                    local_idx = joint.value - arm_start_idx
                    action_np[local_idx] = self.msg.motor_cmd[joint.value].q
                tau = self.arm_ik.solve_tau(action_np)
                for joint in G1_29_JointArmIndex:
                    local_idx = joint.value - arm_start_idx
                    self.msg.motor_cmd[joint.value].tau = tau[local_idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

        return action

    def get_gravity_orientation(self, quaternion):  # get gravity orientation from quaternion
        """Get gravity orientation from quaternion."""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:  # move robot to default position
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        if self.config.is_simulation and self.sim_env is not None:
            self.sim_env.reset()

            for motor in G1_29_JointIndex:
                self.msg.motor_cmd[motor.value].q = default_positions[motor.value]
                self.msg.motor_cmd[motor.value].qd = 0
                self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                self.msg.motor_cmd[motor.value].tau = 0
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
        else:
            total_time = 3.0
            num_steps = int(total_time / control_dt)

            # get current state
            obs = self.get_observation()

            # record current positions
            init_dof_pos = np.zeros(29, dtype=np.float32)
            for motor in G1_29_JointIndex:
                init_dof_pos[motor.value] = obs[f"{motor.name}.q"]

            # Interpolate to default position
            for step in range(num_steps):
                start_time = time.time()

                alpha = step / num_steps
                action_dict = {}
                for motor in G1_29_JointIndex:
                    target_pos = default_positions[motor.value]
                    interp_pos = init_dof_pos[motor.value] * (1 - alpha) + target_pos * alpha
                    action_dict[f"{motor.name}.q"] = float(interp_pos)

                self.send_action(action_dict)

                # Maintain constant control rate
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)

        logger.info("Reached default position")
