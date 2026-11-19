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

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config
from .g1_embodiments import get_g1_embodiment
from .g1_kinematics import G1_29_ArmIK
from .g1_utils import NUM_MOTORS

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize as _SDKChannelFactoryInitialize,
        ChannelPublisher as _SDKChannelPublisher,
        ChannelSubscriber as _SDKChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_ as hg_LowCmd,
        LowState_ as hg_LowState,
    )
    from unitree_sdk2py.utils.crc import CRC
else:
    _SDKChannelFactoryInitialize = None
    _SDKChannelPublisher = None
    _SDKChannelSubscriber = None
    unitree_hg_msg_dds__LowCmd_ = None
    hg_LowCmd = None
    hg_LowState = None
    CRC = None

if TYPE_CHECKING:
    import gymnasium as gym

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
class G1LowState:
    # Keep transport indexing even when some joints are absent from the embodiment.
    motor_state: list[MotorState] = field(default_factory=lambda: [MotorState() for _ in range(NUM_MOTORS)])
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: bytes | None = None  # Raw wireless remote data
    mode_machine: int = 0  # Robot mode


# Preserve imports used by existing G1-29 controllers and downstream integrations.
G1_29_LowState = G1LowState


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config) -> None:
        require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config
        self.embodiment = get_g1_embodiment(config.embodiment)
        self.joint_index = self.embodiment.joint_index
        self.arm_index = self.embodiment.arm_index
        self.control_dt = config.control_dt

        # Initialize cameras config (ZMQ-based) - actual connection in connect()
        self._cameras = make_cameras_from_configs(config.cameras)

        # Import channel classes based on mode
        if config.is_simulation:
            self._ChannelFactoryInitialize = _SDKChannelFactoryInitialize
            self._ChannelPublisher = _SDKChannelPublisher
            self._ChannelSubscriber = _SDKChannelSubscriber
        else:
            from .unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )

            self._ChannelFactoryInitialize = ChannelFactoryInitialize
            self._ChannelPublisher = ChannelPublisher
            self._ChannelSubscriber = ChannelSubscriber

        # Initialize state variables
        self.sim_env = None
        self._env_wrapper: dict[str, dict[int, gym.vector.VectorEnv]] | None = None
        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        # Guards the shared lowcmd message: a controller tick, send_action(), reset() and the
        # shutdown path all publish through it, and a torn update still carries a valid CRC.
        self._lowcmd_lock = threading.Lock()
        # Decides who may drive the joints over a span of time: one controller tick, or a whole
        # reset sweep. Coarser than _lowcmd_lock, which only makes a single command atomic.
        # Reentrant, so a ControlledRobot can hold it across reset() and its controller's reset.
        self._control_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self.subscribe_thread: threading.Thread | None = None

        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        self._ease_in_from: dict[str, float] | None = None
        self._ease_in_start: float | None = None
        # PD gains, resolved in connect() from set_gains() (a controller's) when called, else from the config.
        self._gains: tuple[np.ndarray, np.ndarray] | None = None
        self.kp: np.ndarray
        self.kd: np.ndarray

    def _subscribe_lowstate(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1LowState()

                # Controllers consume the complete 29-slot transport state even when the
                # selected embodiment exposes only a subset as robot features.
                for index in range(NUM_MOTORS):
                    lowstate.motor_state[index].q = msg.motor_state[index].q
                    lowstate.motor_state[index].dq = msg.motor_state[index].dq
                    lowstate.motor_state[index].tau_est = msg.motor_state[index].tau_est
                    lowstate.motor_state[index].temperature = msg.motor_state[index].temperature

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

                with self._lowstate_lock:
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
        with self._lowcmd_lock:
            for motor in self.joint_index:
                key = f"{motor.name}.q"
                if key in action:
                    self.msg.motor_cmd[motor.value].q = action[key]
                    self.msg.motor_cmd[motor.value].dq = 0
                    self.msg.motor_cmd[motor.value].kp = (
                        kp[motor.value] if kp is not None else self.kp[motor.value]
                    )
                    self.msg.motor_cmd[motor.value].kd = (
                        kd[motor.value] if kd is not None else self.kd[motor.value]
                    )
                    self.msg.motor_cmd[motor.value].tau = tau[motor.value] if tau is not None else 0.0

            if len(self.joint_index) < NUM_MOTORS:
                active = {joint.value for joint in self.joint_index}
                for index, command in enumerate(self.msg.motor_cmd):
                    if index not in active:
                        command.mode = 0
                        command.q = command.dq = command.kp = command.kd = command.tau = 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor.name}.q": float for motor in self.joint_index}

    # Low-level control, for the controller a ControlledRobot runs (see lerobot.controllers).

    @property
    def control_lock(self) -> threading.RLock:
        return self._control_lock

    @property
    def passthrough_features(self) -> dict[str, type]:
        """Arm joint targets: callers keep driving the arms under a locomotion controller."""
        return {f"{motor.name}.q": float for motor in self.arm_index}

    def low_level_state(self) -> G1LowState | None:
        """The latest lowstate, with the complete 29-slot transport state."""
        with self._lowstate_lock:
            return self._lowstate

    def write_joint_targets(
        self,
        targets: dict[str, float],
        kp: np.ndarray | list[float] | None = None,
        kd: np.ndarray | list[float] | None = None,
    ) -> None:
        self.publish_lowcmd(targets, kp=kp, kd=kd)

    def set_gains(self, kp: np.ndarray | list[float], kd: np.ndarray | list[float]) -> None:
        """Hold the joints with these gains instead of the config's, from connect() on."""
        self._gains = (np.array(kp, dtype=np.float32), np.array(kd, dtype=np.float32))
        if self.is_connected:
            self.kp, self.kd = self._gains

    def calibrate(self) -> None:
        # TODO: implement g1_29 calibration
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        if self.config.is_simulation and not self.embodiment.supports_simulation:
            raise NotImplementedError(f"Simulation is not implemented for {self.embodiment.name}")

        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            from lerobot.envs import make_env

            self._ChannelFactoryInitialize(0, "lo")
            self._env_wrapper = make_env(self.config.sim_env)
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
        self.subscribe_thread = threading.Thread(target=self._subscribe_lowstate)
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
        deadline = time.time() + 10.0
        while lowstate is None:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for robot state (10s)")
                logger.warning("[UnitreeG1] Waiting for robot state...")
                time.sleep(0.01)
        logger.info("[UnitreeG1] Connected to robot.")
        self.msg.mode_machine = lowstate.mode_machine

        # Prefer the gains set with set_gains() (a controller's: SONIC loads kp/kd from its ONNX);
        # otherwise fall back to the config defaults.
        if self._gains is not None:
            self.kp, self.kd = self._gains
        else:
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)

        for joint in self.joint_index:
            self.msg.motor_cmd[joint].mode = 1
            self.msg.motor_cmd[joint].kp = self.kp[joint.value]
            self.msg.motor_cmd[joint].kd = self.kd[joint.value]
            self.msg.motor_cmd[joint].q = lowstate.motor_state[joint.value].q

        self._arm_ease_in()

    def _send_zero_torque(self) -> None:
        """Send a zero-gain command to make joints passive before shutting down."""
        try:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                return
            action = {f"{motor.name}.q": lowstate.motor_state[motor.value].q for motor in self.joint_index}
            zero_gains = np.zeros(NUM_MOTORS, dtype=np.float32)
            self.publish_lowcmd(action, kp=zero_gains, kd=zero_gains, tau=zero_gains)
            logger.info("Sent zero-torque command for safe shutdown")
        except Exception as e:
            logger.warning(f"Failed to send zero-torque on disconnect: {e}")

    def disconnect(self):
        # Signal threads to stop and unblock any waits. A ControlledRobot has already stopped its
        # controller thread, so zero torque is the robot's last command.
        self._shutdown_event.set()

        # Put robot in passive mode
        if not self.config.is_simulation:
            self._send_zero_torque()

        # Wait for subscribe thread to finish
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not stop cleanly")

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
        with self._lowstate_lock:
            lowstate = self._lowstate
        if lowstate is None:
            return {}

        obs = {}

        # Motors - q, dq, tau for all joints
        for motor in self.joint_index:
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
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if isinstance(cam, DepthCamera) and cam.use_depth:
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        action_to_publish = self._ease_in(action)

        tau: np.ndarray | None = None
        if self.config.gravity_compensation and self.arm_ik is not None:
            tau = np.zeros(NUM_MOTORS, dtype=np.float32)
            action_np = np.array(
                [
                    action_to_publish.get(f"{joint.name}.q", self.msg.motor_cmd[joint.value].q)
                    for joint in self.arm_index
                ],
                dtype=np.float32,
            )
            arm_tau = self.arm_ik.solve_tau(action_np)
            for local_idx, joint in enumerate(self.arm_index):
                tau[joint.value] = arm_tau[local_idx]

        self.publish_lowcmd(action_to_publish, tau=tau)
        return {**action, **action_to_publish}

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        with self._lowstate_lock:
            return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Joint positions for the active joints of the selected embodiment."""
        return {f"{motor.name}.q": float for motor in self.joint_index}

    @property
    def cameras(self) -> dict:
        return self._cameras

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:  # move robot to default position
        if control_dt is None:
            control_dt = self.config.control_dt
        # A ControlledRobot passes its controller's home pose: that is what its policy was trained
        # around (SONIC reads it from the ONNX metadata). The config default is for raw joint teleop.
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        # Hold control authority for the whole sweep. Otherwise a controller thread keeps
        # publishing its own targets throughout, and the robot is driven by two writers at once.
        with self._control_lock:
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.reset()
                self.publish_lowcmd(
                    {f"{motor.name}.q": float(default_positions[motor.value]) for motor in self.joint_index}
                )
            else:
                total_time = 3.0
                num_steps = int(total_time / control_dt)

                # get current state
                obs = self.get_observation()

                # record current positions
                init_dof_pos: np.ndarray = np.zeros(NUM_MOTORS, dtype=np.float32)
                for motor in self.joint_index:
                    init_dof_pos[motor.value] = obs[f"{motor.name}.q"]

                # Interpolate to default position
                for step in range(num_steps):
                    start_time = time.time()

                    alpha = step / num_steps
                    action_dict = {}
                    for motor in self.joint_index:
                        target_pos = default_positions[motor.value]
                        interp_pos = init_dof_pos[motor.value] * (1 - alpha) + target_pos * alpha
                        action_dict[f"{motor.name}.q"] = float(interp_pos)

                    self.publish_lowcmd(action_dict)

                    # Maintain constant control rate
                    elapsed = time.time() - start_time
                    sleep_time = max(0, control_dt - elapsed)
                    time.sleep(sleep_time)

            self._arm_ease_in()

        logger.info("Reached default position")

    def _arm_ease_in(self) -> None:
        if self.config.ease_in_s > 0:
            self._ease_in_from = {}
            self._ease_in_start = None

    def _ease_in(self, action: RobotAction) -> RobotAction:
        """Blend joint targets from the last commanded pose."""
        if self._ease_in_from is None:
            return action

        now = time.monotonic()
        if self._ease_in_start is None:
            # Start on the first action to exclude homing time.
            self._ease_in_start = now
        alpha = (now - self._ease_in_start) / self.config.ease_in_s
        if alpha >= 1.0:
            self._ease_in_from = None
            return action

        eased = dict(action)
        for motor in self.joint_index:
            key = f"{motor.name}.q"
            if key in eased:
                start = self._ease_in_from.setdefault(key, float(self.msg.motor_cmd[motor.value].q))
                eased[key] = (1.0 - alpha) * start + alpha * float(eased[key])
        return eased
