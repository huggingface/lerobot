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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config
from .g1_kinematics import G1_29_ArmIK
from .g1_utils import (
    REMOTE_AXES,
    REMOTE_KEYS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    make_locomotion_controller,
)

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

logger = logging.getLogger(__name__)


@runtime_checkable
class LocomotionController(Protocol):
    control_dt: float

    def run_step(self, action: dict, lowstate) -> dict: ...

    def reset(self) -> None: ...


# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

# Side-channel port on the robot for forwarding exo R3/L3 gripper commands
# (see run_g1_server.gripper_cmd_loop). Real-robot only.
GRIPPER_CMD_PORT = 6002


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
    wireless_remote: bytes | None = None  # Raw wireless remote data
    mode_machine: int = 0  # Robot mode


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config):
        require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config
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
        self._env_wrapper = None
        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None

        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        # Lower-body controller loaded dynamically
        self.controller: LocomotionController | None = make_locomotion_controller(config.controller)

        # Controller thread state
        self._controller_thread = None
        self._controller_action_lock = threading.Lock()
        self.controller_input = default_remote_input()
        self.controller_output = {}

    def _subscribe_lowstate(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # Capture motor states using jointindex
                for joint in G1_29_JointIndex:
                    lowstate.motor_state[joint].q = msg.motor_state[joint].q
                    lowstate.motor_state[joint].dq = msg.motor_state[joint].dq
                    lowstate.motor_state[joint].tau_est = msg.motor_state[joint].tau_est
                    lowstate.motor_state[joint].temperature = msg.motor_state[joint].temperature

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
        for motor in G1_29_JointIndex:
            key = f"{motor.name}.q"
            if key in action:
                self.msg.motor_cmd[motor.value].q = action[key]
                self.msg.motor_cmd[motor.value].qd = 0
                self.msg.motor_cmd[motor.value].kp = (
                    kp[motor.value] if kp is not None else self.kp[motor.value]
                )
                self.msg.motor_cmd[motor.value].kd = (
                    kd[motor.value] if kd is not None else self.kd[motor.value]
                )
                self.msg.motor_cmd[motor.value].tau = tau[motor.value] if tau is not None else 0.0

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
        if self.controller is None:
            return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

        arm_features = {f"{G1_29_JointArmIndex(motor).name}.q": float for motor in G1_29_JointArmIndex}
        remote_features = dict.fromkeys(REMOTE_AXES, float)
        return {**arm_features, **remote_features}

    def _controller_loop(self):
        """Background thread that runs controller at policy's control_dt."""
        control_dt = self.controller.control_dt
        logger.info(f"Controller loop starting with control_dt={control_dt} ({1.0 / control_dt:.1f}Hz)")

        loop_count = 0
        last_log_time = time.time()

        while not self._shutdown_event.is_set():
            start_time = time.time()

            with self._lowstate_lock:
                lowstate = self._lowstate

            if lowstate is not None and self.controller is not None:
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
                controller_action = self.controller.run_step(controller_input, lowstate)

                # Write controller output snapshot
                with self._controller_action_lock:
                    self.controller_output = dict(controller_action)

                ctrl_kp = self.controller.kp if hasattr(self.controller, "kp") else None
                ctrl_kd = self.controller.kd if hasattr(self.controller, "kd") else None
                self.publish_lowcmd(controller_action, kp=ctrl_kp, kd=ctrl_kd)

            elapsed = time.time() - start_time
            sleep_time = max(0, control_dt - elapsed)
            time.sleep(sleep_time)

    def calibrate(self) -> None:
        # TODO: implement g1_29 calibration
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        self._is_disconnected = False
        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            from lerobot.envs import make_env

            self._ChannelFactoryInitialize(0, "lo")
            self._env_wrapper = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            # Extract the actual gym env from the dict structure
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        else:
            self._ChannelFactoryInitialize(0, config=self.config)

        # Gripper command side-channel (real robot only): forwards exo R3/L3 clicks to
        # run_g1_server, which drives the Damiao grippers over CAN.
        self._gripper_sock = None
        self._last_gripper_cmd = None
        self._warned_no_gripper_buttons = False
        if not self.config.is_simulation:
            try:
                import zmq

                sock = zmq.Context.instance().socket(zmq.PUSH)
                sock.setsockopt(zmq.SNDHWM, 2)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(f"tcp://{self.config.robot_ip}:{GRIPPER_CMD_PORT}")
                self._gripper_sock = sock
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Gripper command channel setup failed ({e}); grippers disabled.")
                self._gripper_sock = None

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

        self.kp = np.array(self.config.kp, dtype=np.float32)
        self.kd = np.array(self.config.kd, dtype=np.float32)

        for joint in G1_29_JointIndex:
            self.msg.motor_cmd[joint].mode = 1
            self.msg.motor_cmd[joint].kp = self.kp[joint.value]
            self.msg.motor_cmd[joint].kd = self.kd[joint.value]
            self.msg.motor_cmd[joint].q = lowstate.motor_state[joint.value].q

        # Start controller thread if enabled
        if self.controller is not None:
            # Soft-start: ramp the arms from their current pose to the default position
            # before the locomotion policy takes over, avoiding a jump at startup. The
            # controller owns the legs, so send_action only moves the arms here. Runs in
            # both sim and on the real robot so the ramp is visible in MuJoCo too.
            logger.info("Soft-start: ramping arms to default position...")
            self._interpolate_to_default(duration=3.0)

            self._controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
            self._controller_thread.start()
            fps = int(1.0 / self.controller.control_dt)
            logger.info(f"Controller thread started ({fps}Hz)")

    def _soft_stop(self) -> None:
        """Gently ramp the arms to the default rest pose before shutdown.

        Mirror of the connect-time soft-start. Only runs on the real robot when a
        locomotion controller is active (so the legs stay balanced while the arms
        come down) and lowstate is available to read the current pose.
        """
        if self.config.is_simulation or not self.config.soft_stop:
            return
        if self.controller is None:
            return
        with self._lowstate_lock:
            if self._lowstate is None:
                return
        try:
            logger.info("Soft-stop: ramping arms to default position...")
            self._interpolate_to_default(duration=self.config.soft_stop_duration)
        except Exception as e:
            logger.warning(f"Soft-stop failed ({e}); continuing shutdown.")

    def _send_zero_torque(self) -> None:
        """Send a zero-gain command to make joints passive before shutting down."""
        try:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                return
            action = {f"{motor.name}.q": lowstate.motor_state[motor.value].q for motor in G1_29_JointIndex}
            zero_gains = np.zeros(29, dtype=np.float32)
            self.publish_lowcmd(action, kp=zero_gains, kd=zero_gains, tau=zero_gains)
            logger.info("Sent zero-torque command for safe shutdown")
        except Exception as e:
            logger.warning(f"Failed to send zero-torque on disconnect: {e}")

    def disconnect(self):
        # Idempotent: disconnect() can be called both explicitly and again via GC /
        # interpreter shutdown; re-running soft-stop against already-closed cameras
        # would error, so bail out if we've already torn down.
        if getattr(self, "_is_disconnected", False):
            return
        self._is_disconnected = True

        # Soft-stop: ramp arms slowly back to the rest pose (hands down) while the
        # controller still holds the legs, so they don't drop when we go passive.
        self._soft_stop()

        # Put robot in passive mode before stopping threads
        if not self.config.is_simulation:
            self._send_zero_torque()

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

        # Close gripper command channel
        sock = getattr(self, "_gripper_sock", None)
        if sock is not None:
            sock.close(linger=0)
            self._gripper_sock = None

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
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        action_to_publish = action
        if self.controller is not None:
            # Controller thread owns legs/waist. Here we only update joystick inputs
            # and publish arm targets from the teleoperator.
            self._update_controller_action(action)
            arm_prefixes = tuple(j.name for j in G1_29_JointArmIndex)
            action_to_publish = {
                key: value
                for key, value in action.items()
                if key.endswith(".q") and key.startswith(arm_prefixes)
            }

        tau = None
        if self.config.gravity_compensation and self.arm_ik is not None:
            tau = np.zeros(29, dtype=np.float32)
            action_np = np.array(
                [
                    action_to_publish.get(f"{joint.name}.q", self.msg.motor_cmd[joint.value].q)
                    for joint in G1_29_JointArmIndex
                ],
                dtype=np.float32,
            )
            arm_tau = self.arm_ik.solve_tau(action_np)
            arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value
            for joint in G1_29_JointArmIndex:
                local_idx = joint.value - arm_start_idx
                tau[joint.value] = arm_tau[local_idx]

        self.publish_lowcmd(action_to_publish, tau=tau)
        self._send_gripper_cmd(action)
        return action

    def _send_gripper_cmd(self, action: RobotAction) -> None:
        """Forward exo R3/L3 button flags to run_g1_server to open/close the grippers.

        L3 (left stick, button.4) -> left gripper, R3 (right stick, button.0) -> right.
        Only sends when the state changes to avoid flooding the channel.
        """
        sock = getattr(self, "_gripper_sock", None)
        if sock is None:
            return
        l3 = action.get("remote.button.4")
        r3 = action.get("remote.button.0")
        if l3 is None and r3 is None:
            if not self._warned_no_gripper_buttons:
                logger.warning("[gripper] no remote.button.0/4 in action — teleop not emitting exo buttons")
                self._warned_no_gripper_buttons = True
            return
        cmd = {"L": int(bool(l3)), "R": int(bool(r3))}
        if cmd == self._last_gripper_cmd:
            return
        self._last_gripper_cmd = cmd

        import zmq

        try:
            sock.send_json(cmd, zmq.NOBLOCK)
            logger.info(f"[gripper] sent {cmd} to {self.config.robot_ip}:{GRIPPER_CMD_PORT}")
        except zmq.ZMQError as e:
            logger.warning(f"[gripper] send failed ({e})")

    def _update_controller_action(self, action: RobotAction) -> None:
        """Update controller input state from incoming teleop action."""
        with self._controller_action_lock:
            for key in REMOTE_KEYS:
                if key in action:
                    self.controller_input[key] = action[key]

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        with self._lowstate_lock:
            return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Joint positions for all 29 joints."""
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

    @property
    def cameras(self) -> dict:
        return self._cameras

    def _interpolate_to_default(
        self,
        duration: float = 3.0,
        control_dt: float | None = None,
        default_positions: np.ndarray | list[float] | None = None,
    ) -> None:
        """Smoothly ramp joints from their current pose to the default pose (real robot).

        When a locomotion controller owns the legs, ``send_action`` filters to the arm
        joints, so this effectively ramps only the arms — enough to avoid a startup snap.
        """
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        num_steps = max(1, int(duration / control_dt))

        # record current positions
        obs = self.get_observation()
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
            self.publish_lowcmd(
                {f"{motor.name}.q": float(default_positions[motor.value]) for motor in G1_29_JointIndex}
            )
        else:
            self._interpolate_to_default(
                duration=3.0, control_dt=control_dt, default_positions=default_positions
            )

        # Reset controller internal state (gait phase, obs history, etc.)
        if self.controller is not None and hasattr(self.controller, "reset"):
            self.controller.reset()

        logger.info("Reached default position")
