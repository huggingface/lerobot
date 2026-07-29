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
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config
from .g1_utils import (
    KEYBOARD_KEYS_FIELD,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    lowstate_to_obs,
    make_locomotion_controller,
)

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize as _SDKChannelFactoryInitialize,
        ChannelPublisher as _SDKChannelPublisher,
        ChannelSubscriber as _SDKChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_,
    )
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

# Wireless-remote button byte layout, mapped to the positional button indices the
# locomotion controllers expect. Used in onboard mode to read the physical Unitree
# remote from lowstate (mirrors the exo teleoperator's RemoteController).
_REMOTE_BUTTON_MAP: list[str] = [
    "RB", "LB", "start", "back", "RT", "LT", "", "",
    "A", "B", "X", "Y", "up", "right", "down", "left",
]


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
        #   * client     : thin laptop client. No DDS, no controller. It negotiates a
        #                  controller with ``run_g1_server`` (which runs it onboard),
        #                  PUSHes high-level actions and reads back state + cameras over
        #                  ZMQ. The controller *always* runs on the robot, never here.
        self._client = not config.is_simulation and not config.onboard

        # Initialize cameras config (ZMQ-based) - actual connection in connect()
        self._cameras = make_cameras_from_configs(config.cameras)

        # DDS channel classes are only needed by the in-process control roles. The thin
        # client never touches DDS, so we don't import the socket shim at all.
        if config.is_simulation or config.onboard:
            self._ChannelFactoryInitialize = _SDKChannelFactoryInitialize
            self._ChannelPublisher = _SDKChannelPublisher
            self._ChannelSubscriber = _SDKChannelSubscriber
        else:
            self._ChannelFactoryInitialize = None
            self._ChannelPublisher = None
            self._ChannelSubscriber = None

        # Client-side ZMQ handles / negotiated capabilities (populated in connect()).
        self._client_action_sock = None
        self._client_state_sock = None
        self._client_state_latest: dict[str, float] = {}
        self._client_caps: dict | None = None

        # Initialize state variables
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None

        # Lower-body controller loaded dynamically. GUARDRAIL: the controller must never
        # be built or run on the laptop client -- it always runs onboard (or in sim).
        if self._client:
            self.controller: LocomotionController | None = None
        else:
            self.controller = make_locomotion_controller(config.controller)

            # Token-driven deploy: let a SONIC controller hold a neutral token until the
            # first real one arrives, then hold the last token between control ticks.
            if config.sonic_token_action and hasattr(self.controller, "token_mode"):
                self.controller.token_mode = True

        # Controller thread state
        self._controller_thread = None
        # When set, the controller loop stops publishing low commands so reset() can
        # drive the joints directly without two publishers fighting (single-publisher).
        self._controller_paused = threading.Event()
        self._controller_action_lock = threading.Lock()
        self.controller_input = default_remote_input()
        self.controller_output = {}

        # Onboard-only: parser for the physical Unitree wireless remote (read straight
        # from local lowstate so joystick locomotion works without a laptop round-trip).
        self._joystick = None

        # Token-mode state: last 64-D SONIC latent token commanded by the policy,
        # echoed back as ``observation.state`` so a token-output VLA closes the loop
        # on its own previous token (see ``sonic_token_action``). Seeded to zeros;
        # the controller's startup blend eases joints in regardless.
        self._last_token: np.ndarray | None = None
        if config.sonic_token_action:
            from .controllers.sonic_whole_body import TOKEN_DIM

            self._last_token = np.zeros(TOKEN_DIM, dtype=np.float32)

    def _subscribe_lowstate(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                try:
                    self.sim_env.step()
                except ValueError as e:
                    # Startup race: the sim thread can step once before reset() has
                    # written a valid base pose, giving a zero-norm pelvis quaternion
                    # (scipy>=1.11 raises instead of normalizing). Skip and retry so
                    # the thread survives instead of dying and freezing the sim.
                    if "zero norm" not in str(e).lower():
                        raise
                    time.sleep(self.control_dt)
                    continue

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

    @property
    def _token_state_ft(self) -> dict[str, type]:
        """64-D SONIC latent-token proprio state (``motion_token_state.{i}.pos``).

        Exposed only in ``sonic_token_action`` mode; aggregated by the rollout into a
        64-D ``observation.state`` (the last token the policy commanded).
        """
        if not self.config.sonic_token_action:
            return {}
        from .controllers.sonic_whole_body import TOKEN_DIM, token_state_key

        return {token_state_key(i): float for i in range(TOKEN_DIM)}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._motors_ft,
            **self._token_state_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Role-agnostic: the schema is a pure function of (controller name,
        # sonic_token_action). The thin client advertises the same schema as the
        # onboard robot so the exact same policy output routes straight through.

        # No controller configured at all: raw 29-DoF joint teleop.
        if self.config.controller is None and not self.config.sonic_token_action:
            return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

        # Token-output VLA (SONIC decoder): advertise a 64-D latent-token action space
        # (``motion_token.{i}.pos``) so ``lerobot-rollout`` maps a 64-D policy output
        # straight onto the decoder, bypassing the encoder.
        if self.config.sonic_token_action:
            from .controllers.sonic_whole_body import TOKEN_DIM, token_action_key

            return {token_action_key(i): float for i in range(TOKEN_DIM)}

        # Locomotion controllers (GR00T / Holosoma): arm joint targets + joystick axes.
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

            # Paused during reset() so the reset routine is the sole low-cmd publisher.
            if self._controller_paused.is_set():
                time.sleep(control_dt)
                continue

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

                # Onboard: the physical Unitree remote (in local lowstate) takes
                # priority for locomotion when active; otherwise laptop/ZMQ axes stand.
                if self.config.onboard:
                    wl = self._wireless_remote_input(lowstate)
                    if wl is not None:
                        controller_input.update(wl)

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

    def _wireless_remote_input(self, lowstate) -> dict | None:
        """Parse the physical Unitree remote from lowstate into controller inputs.

        Onboard only. Returns None when the remote is idle so the laptop-provided
        (ZMQ) axes keep control; otherwise the physical remote takes priority.
        """
        js = self._joystick
        if js is None:
            return None
        wr = getattr(lowstate, "wireless_remote", None)
        if not wr or len(wr) < 24:
            return None
        try:
            js.extract(wr)
        except Exception:  # noqa: BLE001
            return None

        axes = {
            "remote.lx": float(js.lx.data),
            "remote.ly": float(js.ly.data),
            "remote.rx": float(js.rx.data),
            "remote.ry": float(js.ry.data),
        }
        active = any(abs(v) > 1e-2 for v in axes.values())
        out = dict(axes)
        for i, name in enumerate(_REMOTE_BUTTON_MAP):
            if name:
                val = float(getattr(js, name).data)
                out[f"remote.button.{i}"] = val
                if val:
                    active = True
        return out if active else None

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

        from .run_g1_server import ACTION_PORT, HANDSHAKE_PORT, STATE_PORT, request_controller

        server_ip = self.config.robot_ip
        if not server_ip:
            raise ValueError("client mode requires config.robot_ip (the G1 running run_g1_server)")

        # 1) Handshake: agree with the server on which controller it will run onboard.
        logger.info(
            "[client] handshaking with %s:%d (controller=%s, token=%s)...",
            server_ip, HANDSHAKE_PORT, self.config.controller, self.config.sonic_token_action,
        )
        self._client_caps = request_controller(
            server_ip,
            self.config.controller,
            sonic_token_action=self.config.sonic_token_action,
            port=HANDSHAKE_PORT,
        )
        logger.info("[client] server agreed: %s", self._client_caps)

        ctx = zmq.Context.instance()

        # 2) Action PUSH: ship compact high-level actions to the onboard controller.
        self._client_action_sock = ctx.socket(zmq.PUSH)
        self._client_action_sock.setsockopt(zmq.SNDHWM, 2)
        self._client_action_sock.setsockopt(zmq.LINGER, 0)
        self._client_action_sock.connect(f"tcp://{server_ip}:{ACTION_PORT}")

        # 3) State SUB: read the onboard observation.state echo (last token / joints).
        self._client_state_sock = ctx.socket(zmq.SUB)
        self._client_state_sock.setsockopt(zmq.CONFLATE, 1)
        self._client_state_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._client_state_sock.connect(f"tcp://{server_ip}:{STATE_PORT}")

        # 4) Cameras (ZMQ ImageServer served by run_g1_server) - same as any client.
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()
        logger.info("[client] connected: actions ->:%d, state <-:%d, %d camera(s).",
                    ACTION_PORT, STATE_PORT, len(self._cameras))

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
        the controller negotiated in the handshake interprets it (token / wb / arm)."""
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
        # Thin-client role: no DDS, no controller. Negotiate the controller with
        # run_g1_server (which runs it onboard), then open the high-level ZMQ links:
        # PUSH actions on :ACTION_PORT, SUB state echo on :STATE_PORT, cameras via ZMQ.
        if self._client:
            self._connect_client()
            return

        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            from lerobot.envs.utils import (
                _download_hub_file,
                _import_hub_module,
                _normalize_hub_result,
            )

            self._ChannelFactoryInitialize(0, "lo")
            # Call the hub env's make_env directly so we can disable the offscreen
            # head_camera renderer. We drive image-conditioned policies from external
            # camera frames, never the sim's own camera, so building a MuJoCo offscreen
            # GL context is pure liability: it
            # crashes with "Failed to make the EGL context current" when GLFW/SDL
            # already own a context, killing the sim thread and hanging on
            # "Waiting for robot state...". publish_images=False -> no renderer.
            repo_id, _, local_file, _ = _download_hub_file(
                "lerobot/unitree-g1-mujoco", True, None
            )
            hub_mod = _import_hub_module(local_file, repo_id)
            raw = hub_mod.make_env(n_envs=1, use_async_envs=False, publish_images=False, cameras=[])
            self._env_wrapper = _normalize_hub_result(raw)
            # Extract the actual gym env from the dict structure
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        elif self.config.onboard:
            # Real robot, controller running onboard against local DDS. Initialize the
            # real SDK channel factory on the robot's DDS interface and take low-level
            # control from the built-in services before we start writing lowcmd.
            if self.config.dds_interface:
                self._ChannelFactoryInitialize(0, self.config.dds_interface)
            else:
                self._ChannelFactoryInitialize(0)
            # Real robot: hand low-level control over from the built-in services.
            # A DDS sim has no MotionSwitcher, so this is skipped there.
            if self.config.release_motion_control:
                self._release_motion_control()
            # Real robot: read the physical wireless remote from lowstate for
            # locomotion. A sim has no physical remote, so leave _joystick=None and
            # let send_action (ZMQ) drive the locomotion axes instead.
            if self.config.physical_remote:
                from unitree_sdk2py.utils.joystick import Joystick

                self._joystick = Joystick()
                for axis in (self._joystick.lx, self._joystick.ly, self._joystick.rx, self._joystick.ry):
                    axis.smooth = 1.0
                    axis.deadzone = 0.0
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

        self.kp = np.array(self.config.kp, dtype=np.float32)
        self.kd = np.array(self.config.kd, dtype=np.float32)
        if self.controller is not None and hasattr(self.controller, "kp"):
            self.kp = np.array(self.controller.kp, dtype=np.float32)
            self.kd = np.array(self.controller.kd, dtype=np.float32)

        for joint in G1_29_JointIndex:
            self.msg.motor_cmd[joint].mode = 1
            self.msg.motor_cmd[joint].kp = self.kp[joint.value]
            self.msg.motor_cmd[joint].kd = self.kd[joint.value]
            self.msg.motor_cmd[joint].q = lowstate.motor_state[joint.value].q

        # Start controller thread if enabled. Skipped when run_controller_thread is
        # False so a caller can step the controller synchronously (faithful replay).
        if self.controller is not None and self.config.run_controller_thread:
            self._controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
            self._controller_thread.start()
            fps = int(1.0 / self.controller.control_dt)
            logger.info(f"Controller thread started ({fps}Hz)")
        elif self.controller is not None:
            logger.info("Controller thread disabled (run_controller_thread=False); "
                        "caller must drive controller.run_step synchronously.")

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

    def _graceful_stop(self) -> None:
        """Soft shutdown: hold the current pose and ramp joint stiffness (kp) to zero
        over ``graceful_stop_s`` while keeping damping (kd), then go passive.

        Prevents the robot from collapsing the instant control ends (a bare
        zero-torque command is kp=kd=0 ≈ free-fall). Must run after the controller
        loop has stopped so the two aren't publishing at once.
        """
        if self.config.graceful_stop_s <= 0:
            self._send_zero_torque()
            return
        with self._lowstate_lock:
            lowstate = self._lowstate
        if lowstate is None:
            self._send_zero_torque()
            return
        q_hold = {f"{motor.name}.q": lowstate.motor_state[motor.value].q for motor in G1_29_JointIndex}
        kp = np.array(self.kp, dtype=np.float32)
        kd = np.array(self.kd, dtype=np.float32)
        zeros = np.zeros(29, dtype=np.float32)
        dt = self.controller.control_dt if self.controller is not None else self.config.control_dt
        steps = max(1, int(self.config.graceful_stop_s / dt))
        logger.info("Graceful stop: damping down over %.1fs", self.config.graceful_stop_s)
        for i in range(steps):
            ratio = (i + 1) / steps
            self.publish_lowcmd(q_hold, kp=kp * (1.0 - ratio), kd=kd, tau=zeros)
            time.sleep(dt)
        self._send_zero_torque()

    def disconnect(self):
        if self._client:
            self._disconnect_client()
            return

        # Stop the controller loop first so it isn't fighting the shutdown ramp.
        self._shutdown_event.set()
        controller_stopped = True
        if self._controller_thread is not None:
            # Wait long enough for any in-flight inference tick to finish and the loop
            # to observe the shutdown flag, so no stray low command is published while
            # the ramp runs (the shutdown routine must be the single publisher).
            self._controller_thread.join(timeout=5.0)
            if self._controller_thread.is_alive():
                controller_stopped = False
                logger.error(
                    "Controller thread did not stop; skipping graceful ramp to avoid "
                    "concurrent low commands (fail-safe: joints keep last command until exit)"
                )

        # Soft, damped settle instead of an instant limp (real robot only; the
        # subscribe thread is still alive here to supply the current pose). Only ramp
        # once the controller thread has definitely exited.
        if not self.config.is_simulation and controller_stopped:
            self._graceful_stop()

        if self.controller is not None and hasattr(self.controller, "shutdown"):
            self.controller.shutdown()

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
        if self._client:
            return self._get_observation_client()

        with self._lowstate_lock:
            lowstate = self._lowstate
        if lowstate is None:
            return {}

        # Motors + IMU + wireless remote (shared lowstate -> obs mapping)
        obs = lowstate_to_obs(lowstate)

        # Token mode: echo the last commanded latent token as observation.state so a
        # token-output VLA closes the loop on its own previous token.
        if self.config.sonic_token_action:
            from .controllers.sonic_whole_body import token_state_key

            token = self._last_token if self._last_token is not None else []
            for i, v in enumerate(token):
                obs[token_state_key(i)] = float(v)

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
            if self.config.sonic_token_action:
                from .controllers.sonic_whole_body import _extract_token_from_action

                token = _extract_token_from_action(action)
                if token is not None:
                    self._last_token = token
            self._update_controller_action(action)
            if getattr(self.controller, "full_body", False):
                return action
            # Controller thread owns legs/waist. Here we only update joystick inputs
            # and publish arm targets from the teleoperator.
            arm_prefixes = tuple(j.name for j in G1_29_JointArmIndex)
            action_to_publish = {
                key: value
                for key, value in action.items()
                if key.endswith(".q") and key.startswith(arm_prefixes)
            }

        self.publish_lowcmd(action_to_publish)
        return action

    def _update_controller_action(self, action: RobotAction) -> None:
        """Update controller input state from an incoming teleop action.

        Controller-agnostic: every value-carrying key is forwarded verbatim into
        ``controller_input`` (whole-body ``wb.{i}.pos`` from a 34-D VLA, or whatever a
        future controller expects), and each controller extracts only the keys it
        understands. The robot deliberately does not enumerate any controller's key
        schema here.

        KeyboardTeleop is the one special case: it emits the currently-pressed keys as
        bare action keys with a ``None`` value (``dict.fromkeys(pressed, None)``), so
        those are collected into a single held-key set under ``KEYBOARD_KEYS_FIELD``,
        rebuilt each tick so releases clear. Special keys arrive as pynput objects and
        are normalised to their name ("space", ...).
        """
        with self._controller_action_lock:
            self.controller_input[KEYBOARD_KEYS_FIELD] = {
                (k if isinstance(k, str) else getattr(k, "name", str(k)))
                for k, value in action.items()
                if value is None
            }
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

        # Full-body controllers (SONIC / OpenHLM) own the whole 29-DoF command and
        # ignore ``<joint>.q`` in send_action(), so reset() must publish the default
        # pose directly. Pause the background controller first so the two aren't both
        # writing low commands while the robot moves to the default pose.
        full_body = getattr(self.controller, "full_body", False)
        paused = False
        if full_body and self._controller_thread is not None:
            self._controller_paused.set()
            paused = True
            time.sleep(control_dt)  # let any in-flight controller tick settle

        try:
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

                    # Full-body controllers no-op in send_action(); publish the pose
                    # directly (arm-only controllers keep the send_action() path).
                    if full_body:
                        self.publish_lowcmd(action_dict)
                    else:
                        self.send_action(action_dict)

                    # Maintain constant control rate
                    elapsed = time.time() - start_time
                    sleep_time = max(0, control_dt - elapsed)
                    time.sleep(sleep_time)

            # Reset controller internal state (gait phase, obs history, etc.) before
            # resuming so its buffers reflect the post-reset pose.
            if self.controller is not None and hasattr(self.controller, "reset"):
                self.controller.reset()
        finally:
            if paused:
                self._controller_paused.clear()

        logger.info("Reached default position")
