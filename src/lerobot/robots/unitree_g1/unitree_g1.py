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

import importlib
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
from lerobot.robots.unitree_g1.g1_utils import (
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)
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

        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        # Lower-body controller loaded dynamically from robots/unitree_g1/controller
        self.controller = None
        if config.controller:
            controller_cls = getattr(
                importlib.import_module("lerobot.robots.unitree_g1.controller"),
                config.controller,
                None,
            ) or (_ for _ in ()).throw(ValueError(f"Unknown controller: {config.controller}"))
            self.controller = controller_cls()

        # Controller thread state
        self._controller_thread = None
        self._controller_action_lock = threading.Lock()
        self.controller_input = {
            "remote.lx": 0.0,
            "remote.ly": 0.0,
            "remote.rx": 0.0,
            "remote.ry": 0.0,
        }
        self.controller_output = {}

    def subscribe_lowstate(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

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

    def publish_lowcmd(
        self,
        action: RobotAction,
        kp: np.ndarray | list[float] | None = None,
        kd: np.ndarray | list[float] | None = None,
        tau: np.ndarray | list[float] | None = None,
    ) -> None:  # writes robot command whenever requested
        for motor in G1_29_JointIndex:
            key = f"{motor.name}.q"
            if key in action:
                self.msg.motor_cmd[motor.value].q = action[key]
                self.msg.motor_cmd[motor.value].qd = 0
                self.msg.motor_cmd[motor.value].kp = kp[motor.value] if kp is not None else self.kp[motor.value]
                self.msg.motor_cmd[motor.value].kd = kd[motor.value] if kd is not None else self.kd[motor.value]
                self.msg.motor_cmd[motor.value].tau = tau[motor.value] if tau is not None else 0.0

        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)

    def _controller_loop(self):
        """Background thread that runs controller at policy's control_dt."""
        control_dt = self.controller.control_dt
        logger.info(f"Controller loop starting with control_dt={control_dt} ({1.0 / control_dt:.1f}Hz)")

        loop_count = 0
        last_log_time = time.time()

        while not self._shutdown_event.is_set():
            start_time = time.time()

            if self._lowstate is not None and self.controller is not None:
                loop_count += 1
                if time.time() - last_log_time >= 5.0:  # Log every 5 seconds
                    actual_hz = loop_count / (time.time() - last_log_time)
                    logger.info(
                        f"Controller actual rate: {actual_hz:.1f}Hz (target: {1.0 / control_dt:.1f}Hz)"
                    )
                    loop_count = 0
                    last_log_time = time.time()
                # Read controller input snapshot
                with self._controller_action_lock:
                    controller_input = dict(self.controller_input)

                # Run controller step
                controller_action = self.controller.run_step(controller_input, self._lowstate)

                # Write controller output snapshot
                with self._controller_action_lock:
                    self.controller_output = dict(controller_action)

                ctrl_kp = self.controller.kp if hasattr(self.controller, "kp") else None
                ctrl_kd = self.controller.kd if hasattr(self.controller, "kd") else None
                self.publish_lowcmd(controller_action, kp=ctrl_kp, kd=ctrl_kd)

            elapsed = time.time() - start_time
            sleep_time = max(0, control_dt - elapsed)
            time.sleep(sleep_time)

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.controller is None:
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
        self.subscribe_thread = threading.Thread(target=self.subscribe_lowstate)
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

        # Initialize kp/kd from controller for legs/waist, config for arms
        if self.controller is not None and hasattr(self.controller, "kp"):
            # Use controller gains for legs/waist (0-14), config gains for arms (15-28)
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)
            # Override legs and waist with controller gains
            self.kp[:15] = self.controller.kp[:15]
            self.kd[:15] = self.controller.kd[:15]
            logger.info("Using KP/KD from controller (legs/waist) + config (arms)")
            logger.info(f"  Legs KP: {self.kp[:12].tolist()}")
            logger.info(f"  Arms KP: {self.kp[15:].tolist()}")
        else:
            # Use default from config
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)
            logger.info("Using KP/KD from config")

        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp[id.value]
            self.msg.motor_cmd[id].kd = self.kd[id.value]
            self.msg.motor_cmd[id].q = lowstate.motor_state[id.value].q

        # Start controller thread if enabled
        if self.controller is not None:
            self._controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
            self._controller_thread.start()
            fps = int(1.0 / self.controller.control_dt)
            logger.info(f"Controller thread started ({fps}Hz)")

    def disconnect(self):
        # Signal thread to stop and unblock any waits
        self._shutdown_event.set()

        # Wait for subscribe thread to finish
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not stop cleanly")

        # Wait for controller thread to finish
        if self._controller_thread is not None:
            self._controller_thread.join(timeout=2.0)
            if self._controller_thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")

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
            obs[cam_name] = cam.read_latest()

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

    def _update_controller_action(self, action: RobotAction) -> None:
        """Update controller input state from incoming teleop action."""
        with self._controller_action_lock:
            for key in ("remote.lx", "remote.ly", "remote.rx", "remote.ry"):
                if key in action:
                    self.controller_input[key] = action[key]
            for i in range(16):
                btn_key = f"remote.button.{i}"
                if btn_key in action:
                    self.controller_input[btn_key] = action[btn_key]

    def send_action(self, action: RobotAction) -> RobotAction:
        action_to_publish = action
        if self.controller is not None:
            # Controller thread owns legs/waist. Here we only update joystick inputs
            # and publish arm targets from the teleoperator.
            self._update_controller_action(action)
            action_to_publish = {
                key: value
                for key, value in action.items()
                if key.endswith(".q") and key.startswith(tuple(f"{j.name}" for j in G1_29_JointArmIndex))
            }

        tau = None
        if self.config.gravity_compensation and self.arm_ik is not None:
            tau = np.zeros(29, dtype=np.float32)
            action_np = np.array(
                [action_to_publish.get(f"{joint.name}.q", self.msg.motor_cmd[joint.value].q) for joint in G1_29_JointArmIndex],
                dtype=np.float32,
            )
            arm_tau = self.arm_ik.solve_tau(action_np)
            arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value
            for joint in G1_29_JointArmIndex:
                local_idx = joint.value - arm_start_idx
                tau[joint.value] = arm_tau[local_idx]

        self.publish_lowcmd(action_to_publish, tau=tau)
        return action

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
            self.publish_lowcmd({f"{motor.name}.q": float(default_positions[motor.value]) for motor in G1_29_JointIndex})
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
