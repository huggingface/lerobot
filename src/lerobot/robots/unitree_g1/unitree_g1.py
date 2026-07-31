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

import contextlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config
from .g1_kinematics import G1_29_ArmIK
from .g1_utils import (
    REMOTE_AXES,
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

        # Three mutually-exclusive roles:
        #   * simulation : local DDS + controller run in-process against a MuJoCo world.
        #   * onboard    : local DDS + controller run in-process on the robot NX.
        #   * client     : thin laptop client. No DDS, no controller. It PUSHes high-level
        #                  actions to ``run_g1_server --onboard`` and reads back state +
        #                  cameras over ZMQ. The controller *always* runs on the robot.
        self._client = not config.is_simulation and not config.onboard

        # Initialize cameras config (ZMQ-based) - actual connection in connect()
        self._cameras = make_cameras_from_configs(config.cameras)

        # DDS channels are only needed by the in-process control roles (sim / onboard),
        # which both drive the real Unitree SDK. The thin client never touches DDS.
        if config.is_simulation or config.onboard:
            self._ChannelFactoryInitialize = _SDKChannelFactoryInitialize
            self._ChannelPublisher = _SDKChannelPublisher
            self._ChannelSubscriber = _SDKChannelSubscriber
        else:
            self._ChannelFactoryInitialize = None
            self._ChannelPublisher = None
            self._ChannelSubscriber = None

        # Client-side ZMQ handles (populated in connect()).
        self._client_action_sock = None
        self._client_state_sock = None
        self._client_state_latest: dict[str, float] = {}

        # Initialize state variables
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None

        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        # Controller loaded dynamically. GUARDRAIL: the controller must never be built or
        # run on the laptop client -- it always runs onboard (or in sim).
        if self._client:
            self.controller: LocomotionController | None = None
        else:
            self.controller = make_locomotion_controller(config.controller)
        # Controller thread state
        self._controller_thread = None
        self._controller_action_lock = threading.Lock()
        self.controller_input = default_remote_input()
        self.controller_output = {}

    @property
    def _sonic_token(self) -> bool:
        """Whether the SONIC whole-body decoder is active.

        A SONIC controller consumes a 64-D latent motion token as its action and echoes
        the last commanded token as ``observation.state``. Keyed purely off the selected
        controller so the token interface is implicit -- no separate config flag, and the
        thin client (which has no controller instance) can still advertise the schema.
        """
        return self.config.controller == "SonicWholeBodyController"

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
        # Controllers may contribute their own proprio features (e.g. SONIC's token state).
        # The thin client has no controller instance, so mirror the onboard token schema
        # by controller name (SONIC echoes its last token as observation.state).
        controller_ft = getattr(self.controller, "observation_ft", {})
        if self._client and self._sonic_token:
            from .controllers.sonic_whole_body import TOKEN_DIM, TOKEN_STATE_PREFIX

            controller_ft = {f"{TOKEN_STATE_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}
        return {**self._motors_ft, **controller_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Role-agnostic: the schema is a pure function of the configured controller name,
        # so the thin client advertises the same action space as the onboard robot.

        # No controller configured at all: raw 29-DoF joint teleop.
        if self.config.controller is None:
            return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

        # Whole-body controllers (SONIC): 64-D latent token. On the thin client there is
        # no controller instance, so advertise the same token schema by controller name.
        controller_ft = getattr(self.controller, "action_ft", None)
        if controller_ft is not None:
            return dict(controller_ft)
        if self._client and self._sonic_token:
            from .controllers.sonic_whole_body import TOKEN_ACTION_PREFIX, TOKEN_DIM

            return {f"{TOKEN_ACTION_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}

        # Locomotion controllers (GR00T / Holosoma): arm joint targets + joystick axes.
        # TODO: have GR00T/Holosoma advertise their own action_features too, so every
        # controller declares its action space and this fallthrough can be dropped.
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

    def _release_motion_control(self) -> None:
        """Release the robot's built-in motion services so we can send raw lowcmd.

        Onboard-only. Mirrors run_g1_server.py: on the real robot the factory
        locomotion/hand services must relinquish control before our controller can
        write to ``rt/lowcmd``, otherwise commands are ignored or fought.
        """
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        while result is not None and "name" in result and result["name"]:
            logger.info("[UnitreeG1] Releasing built-in mode '%s'...", result["name"])
            msc.ReleaseMode()
            _, result = msc.CheckMode()
            time.sleep(1.0)

    # ------------------------------------------------------------------ #
    # Thin-client role (laptop): no DDS, no controller. Talks to run_g1_server
    # over ZMQ. The controller ALWAYS runs onboard; we only relay high-level
    # actions and read back the state echo + camera frames.
    # ------------------------------------------------------------------ #
    def _connect_client(self) -> None:
        import zmq

        from .run_g1_server import ACTION_PORT, STATE_PORT

        server_ip = self.config.robot_ip
        if not server_ip:
            raise ValueError("client mode requires config.robot_ip (the G1 running run_g1_server)")

        ctx = zmq.Context.instance()

        # Action PUSH: ship compact high-level actions to the onboard controller. The
        # server runs the controller selected by its own ``--controller`` flag; both
        # sides use ``config.controller`` to agree on the action schema (no handshake).
        self._client_action_sock = ctx.socket(zmq.PUSH)
        self._client_action_sock.setsockopt(zmq.SNDHWM, 2)
        self._client_action_sock.setsockopt(zmq.LINGER, 0)
        self._client_action_sock.connect(f"tcp://{server_ip}:{ACTION_PORT}")

        # State SUB: read the onboard observation.state echo (last token / joints).
        self._client_state_sock = ctx.socket(zmq.SUB)
        self._client_state_sock.setsockopt(zmq.CONFLATE, 1)
        self._client_state_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._client_state_sock.connect(f"tcp://{server_ip}:{STATE_PORT}")

        # Cameras (ZMQ ImageServer served by run_g1_server) - same as any client.
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()
        logger.info(
            "[client] connected to %s: actions ->:%d, state <-:%d, %d camera(s).",
            server_ip,
            ACTION_PORT,
            STATE_PORT,
            len(self._cameras),
        )

    def _recv_client_state(self) -> None:
        """Drain the state SUB (CONFLATE keeps only the freshest) into the latest cache."""
        import zmq

        if self._client_state_sock is None:
            return
        while True:
            try:
                state = self._client_state_sock.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except (ValueError, zmq.ZMQError):
                break
            if isinstance(state, dict):
                self._client_state_latest = {k: float(v) for k, v in state.items()}

    def _get_observation_client(self) -> RobotObservation:
        self._recv_client_state()
        obs: dict = dict(self._client_state_latest)
        for cam_name, cam in self._cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()
        return obs

    def _send_action_client(self, action: RobotAction) -> RobotAction:
        """Relay the raw action straight to the onboard controller. NO processing here:
        the onboard controller interprets it (token / wb / arm)."""
        import zmq

        if self._client_action_sock is None:
            raise DeviceNotConnectedError("UnitreeG1 client is not connected")
        payload = json.dumps({k: float(v) for k, v in action.items()}).encode("utf-8")
        with contextlib.suppress(zmq.Again):
            self._client_action_sock.send(payload, zmq.NOBLOCK)
        return action

    def _disconnect_client(self) -> None:
        for sock in (self._client_action_sock, self._client_state_sock):
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close(linger=0)
        self._client_action_sock = None
        self._client_state_sock = None
        for cam in self._cameras.values():
            with contextlib.suppress(Exception):
                cam.disconnect()

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        # Thin-client role: no DDS, no controller. Open the high-level ZMQ links to
        # run_g1_server --onboard (which runs the controller): PUSH actions on
        # :ACTION_PORT, SUB state echo on :STATE_PORT, cameras via ZMQ.
        if self._client:
            self._connect_client()
            return

        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            from lerobot.envs import make_env

            self._ChannelFactoryInitialize(0, "lo")
            self._env_wrapper = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            # Extract the actual gym env from the dict structure
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        elif self.config.onboard:
            # Real robot, controller running onboard against local DDS. Initialize the
            # real SDK channel factory, then take low-level control from the built-in
            # motion services before we start writing lowcmd.
            self._ChannelFactoryInitialize(0)
            self._release_motion_control()

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

        # Prefer the active controller's gains (e.g. SONIC loads kp/kd from its ONNX);
        # otherwise fall back to the config defaults.
        if self.controller is not None and hasattr(self.controller, "kp"):
            self.kp = np.array(self.controller.kp, dtype=np.float32)
            self.kd = np.array(self.controller.kd, dtype=np.float32)
        else:
            self.kp = np.array(self.config.kp, dtype=np.float32)
            self.kd = np.array(self.config.kd, dtype=np.float32)

        for joint in G1_29_JointIndex:
            self.msg.motor_cmd[joint].mode = 1
            self.msg.motor_cmd[joint].kp = self.kp[joint.value]
            self.msg.motor_cmd[joint].kd = self.kd[joint.value]
            self.msg.motor_cmd[joint].q = lowstate.motor_state[joint.value].q

        # Ease into the controller's home pose before it takes over, so the first commands
        # don't snap from the connect-time pose.
        if self.controller is not None and hasattr(self.controller, "default_angles"):
            self.reset(default_positions=self.controller.default_angles)

        # Start controller thread if enabled
        if self.controller is not None:
            self._controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
            self._controller_thread.start()
            fps = int(1.0 / self.controller.control_dt)
            logger.info(f"Controller thread started ({fps}Hz)")

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
        if self._client:
            self._disconnect_client()
            return

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

        # Disconnect cameras
        for cam in self._cameras.values():
            cam.disconnect()

    def get_observation(self) -> RobotObservation:
        if self._client:
            return self._get_observation_client()

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

        # Controller-contributed observation (e.g. SONIC echoes its last decoded token as
        # observation.state so a token-output VLA closes the loop on its own previous token).
        if self.controller is not None and hasattr(self.controller, "observation_state"):
            obs.update(self.controller.observation_state())

        # Cameras - read images from ZMQ cameras
        for cam_name, cam in self._cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._client:
            return self._send_action_client(action)

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
        return action

    def _update_controller_action(self, action: RobotAction) -> None:
        """Forward incoming teleop action values into ``controller_input``; each controller
        reads only the keys it understands."""
        with self._controller_action_lock:
            for key, value in action.items():
                if isinstance(key, str) and value is not None:
                    self.controller_input[key] = value

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        if self._client:
            return self._client_action_sock is not None
        with self._lowstate_lock:
            return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Joint positions for all 29 joints."""
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

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
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        if self.config.is_simulation and self.sim_env is not None:
            self.sim_env.reset()
            self.publish_lowcmd(
                {f"{motor.name}.q": float(default_positions[motor.value]) for motor in G1_29_JointIndex}
            )
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

                self.publish_lowcmd(action_dict)

                # Maintain constant control rate
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)

        # Reset controller internal state (gait phase, obs history, etc.)
        if self.controller is not None and hasattr(self.controller, "reset"):
            self.controller.reset()

        logger.info("Reached default position")
