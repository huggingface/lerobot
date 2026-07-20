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
    KEYBOARD_KEYS_FIELD,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    lowstate_to_obs,
    make_locomotion_controller,
    obs_to_wb34_state,
)

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize as _SDKChannelFactoryInitialize,
        ChannelPublisher as _SDKChannelPublisher,
        ChannelSubscriber as _SDKChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__HandCmd_ as hg_HandCmd_default,
        unitree_hg_msg_dds__LowCmd_,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        HandCmd_ as hg_HandCmd,
        LowCmd_ as hg_LowCmd,
        LowState_ as hg_LowState,
    )
    from unitree_sdk2py.utils.crc import CRC
else:
    _SDKChannelFactoryInitialize = None
    _SDKChannelPublisher = None
    _SDKChannelSubscriber = None
    unitree_hg_msg_dds__LowCmd_ = None
    hg_HandCmd_default = None
    hg_HandCmd = None
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

        # Replay-camera state (decoded frames per robot camera name + play cursor).
        self._replay_frames: dict[str, list[np.ndarray]] = {}
        self._replay_len = 0
        self._replay_idx = 0
        if config.replay_camera_parquet and config.replay_camera_map:
            self._load_replay_frames()

    def _load_replay_frames(self) -> None:
        """Decode recorded episode frames from a parquet into per-camera image lists."""
        import io

        import pyarrow.parquet as pq
        from PIL import Image

        table = pq.read_table(self.config.replay_camera_parquet)
        cols = {col: table.column(col).to_pylist() for col in self.config.replay_camera_map.values()}
        self._replay_len = table.num_rows

        def decode(cell) -> np.ndarray:
            data = cell["bytes"] if isinstance(cell, dict) else cell
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)

        for cam_name, column in self.config.replay_camera_map.items():
            self._replay_frames[cam_name] = [decode(c) for c in cols[column]]
        logger.info(
            "Loaded %d replay frames for cameras %s from %s",
            self._replay_len,
            list(self.config.replay_camera_map),
            self.config.replay_camera_parquet,
        )

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

    @property
    def _wb_state_ft(self) -> dict[str, type]:
        """34-D whole-body proprio state (``wb_state.{i}.pos``) for dense controllers.

        Exposed only when the controller consumes a dense whole-body command
        (OpenHLM / pi0.5). These ``.pos`` scalars are aggregated by the rollout
        pipeline into a single 34-D ``observation.state`` for the policy.
        """
        if not getattr(self.controller, "wb_action", False):
            return {}
        from .g1_utils import WB_ACTION_DIM

        return {f"wb_state.{i}.pos": float for i in range(WB_ACTION_DIM)}

    @property
    def _empty_cameras_ft(self) -> dict[str, tuple]:
        """Synthetic zero-image cameras (see ``UnitreeG1Config.empty_cameras``)."""
        h, w = self.config.empty_camera_hw
        return dict.fromkeys(self.config.empty_cameras, (h, w, 3))

    @property
    def _replay_cameras_ft(self) -> dict[str, tuple]:
        """Replay cameras, shaped from their first decoded frame."""
        return {name: frames[0].shape for name, frames in self._replay_frames.items() if frames}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._motors_ft,
            **self._wb_state_ft,
            **self._empty_cameras_ft,
            **self._replay_cameras_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.controller is None:
            return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

        # Dense whole-body controllers (SONIC / OpenHLM, pi0.5) consume a single
        # 34-D command per tick. Expose it as ``wb.{i}.pos`` joint-position features
        # so ``lerobot-rollout`` maps a 34-D policy output straight onto the robot.
        if getattr(self.controller, "wb_action", False):
            from .g1_utils import WB_ACTION_DIM, wb_action_key

            return {wb_action_key(i): float for i in range(WB_ACTION_DIM)}

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
        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            from lerobot.envs import make_env

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

        # Dex3 hand command publishers (grasping). Driven by the OpenHLM grip scalars.
        self._hand_publishers = {}
        if self.config.publish_hands:
            self._left_hand_cmd = hg_HandCmd_default()
            self._right_hand_cmd = hg_HandCmd_default()
            self._hand_publishers["left"] = self._ChannelPublisher("rt/dex3/left/cmd", hg_HandCmd)
            self._hand_publishers["right"] = self._ChannelPublisher("rt/dex3/right/cmd", hg_HandCmd)
            for pub in self._hand_publishers.values():
                pub.Init()
            logger.info("Dex3 hand command publishers initialized (rt/dex3/{left,right}/cmd)")

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
        # Stop the controller loop first so it isn't fighting the shutdown ramp.
        self._shutdown_event.set()
        if self._controller_thread is not None:
            self._controller_thread.join(timeout=2.0)
            if self._controller_thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")

        # Soft, damped settle instead of an instant limp (real robot only; the
        # subscribe thread is still alive here to supply the current pose).
        if not self.config.is_simulation:
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
        with self._lowstate_lock:
            lowstate = self._lowstate
        if lowstate is None:
            return {}

        # Motors + IMU + wireless remote (shared lowstate -> obs mapping)
        obs = lowstate_to_obs(lowstate)

        # Dense whole-body controllers (OpenHLM / pi0.5): expose the 34-D proprio
        # state as ``wb_state.{i}.pos`` so the rollout aggregates it into
        # ``observation.state`` for the policy.
        if getattr(self.controller, "wb_action", False):
            wb_state = obs_to_wb34_state(obs)
            for i, v in enumerate(wb_state):
                obs[f"wb_state.{i}.pos"] = float(v)

        # Synthetic empty cameras: black frames so image-conditioned policies run
        # before real cameras are wired.
        if self.config.empty_cameras:
            h, w = self.config.empty_camera_hw
            black = np.zeros((h, w, 3), dtype=np.uint8)
            for name in self.config.empty_cameras:
                obs[name] = black

        # Replay cameras: serve the current recorded frame per camera, then advance.
        if self._replay_len:
            idx = self._replay_idx
            if idx >= self._replay_len:
                idx = self._replay_len - 1 if not self.config.replay_camera_loop else idx % self._replay_len
            for name, frames in self._replay_frames.items():
                obs[name] = frames[idx]
            self._replay_idx += 1

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
            self._update_controller_action(action)
            if self.config.publish_hands and getattr(self.controller, "wb_action", False):
                self._publish_hand_cmds(action)
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

    def _publish_hand_cmds(self, action: RobotAction) -> None:
        """Drive the Dex3 hands from the OpenHLM grip scalars in a 34-D wb action.

        ``wb.7.pos`` is the left grip and ``wb.15.pos`` the right grip. Each scalar in
        [0, 1] (``hand_open_grip_value`` == fully open) is turned into a curl amount and
        scaled onto ``hand_closed_pose`` (7 joints), then published as a PD target on
        ``rt/dex3/{left,right}/cmd`` so the fingers close when the policy grips.
        """
        if not self._hand_publishers:
            return
        from .g1_utils import wb_action_key

        open_val = float(self.config.hand_open_grip_value)
        closed_val = float(self.config.hand_closed_grip_value)
        closed_pose = self.config.hand_closed_pose
        kp, kd = float(self.config.hand_kp), float(self.config.hand_kd)
        span = (closed_val - open_val) or 1.0

        def curl_amount(grip: float) -> float:
            # Fraction of the way from the open scalar to the closed scalar, in [0, 1].
            return float(min(max((grip - open_val) / span, 0.0), 1.0))

        for side, grip_idx, cmd in (
            ("left", 7, self._left_hand_cmd),
            ("right", 15, self._right_hand_cmd),
        ):
            grip = action.get(wb_action_key(grip_idx))
            if grip is None:
                continue
            amount = curl_amount(float(grip))
            for i, closed_q in enumerate(closed_pose):
                cmd.motor_cmd[i].q = float(closed_q) * amount
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kp = kp
                cmd.motor_cmd[i].kd = kd
                cmd.motor_cmd[i].tau = 0.0
            self._hand_publishers[side].Write(cmd)

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

                self.send_action(action_dict)

                # Maintain constant control rate
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)

        # Reset controller internal state (gait phase, obs history, etc.)
        if self.controller is not None and hasattr(self.controller, "reset"):
            self.controller.reset()

        logger.info("Reached default position")
