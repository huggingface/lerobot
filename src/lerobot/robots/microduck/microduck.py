# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import requests

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_microduck import MicroDuckConfig
from .rpc import DuckRpc

logger = logging.getLogger(__name__)

# Wire protocol version this client was written against (duck-ipc-proto `API_VERSION`).
API_VERSION = 41

# `duck-ipc-proto::JOINT_NAMES`, the order of `joints` / `targets` in robot.state.
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

# Order of `robot.head` params and of the `head` array in robot.state.
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

TWIST_ACTIONS = ("vx", "vy", "vyaw")
HEAD_ACTIONS = tuple(f"{name}.pos" for name in HEAD_JOINTS)

CAMERA = "front"


class MicroDuck(Robot):
    """Pollen Robotics Microduck, a small biped walking with an onboard RL policy.

    Actions are the policy's intents: a body twist (``vx``, ``vy`` in m/s, ``vyaw`` in rad/s, trunk frame,
    x forward, y left, positive yaw turns left) and a head pose (rad). They are sent to ``robotd`` as
    ``robot.move`` / ``robot.head`` notifications on every ``send_action``; robotd zeroes the twist if no
    intent arrives for 500 ms, so the control loop must run at 2 Hz or more.

    Observations come from robotd's ``robot.state`` stream (joints, IMU, odometry, applied twist) and from
    mediad's snapshot endpoint for the head camera.
    """

    config_class = MicroDuckConfig
    name = "microduck"

    def __init__(self, config: MicroDuckConfig):
        super().__init__(config)
        self.config = config

        self._cmd = DuckRpc(config.robotd_address, timeout_s=config.rpc_timeout_s)
        self._state_rpc = DuckRpc(config.robotd_address, timeout_s=config.rpc_timeout_s)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        self._state_lock = threading.Lock()
        self._latest_state: dict[str, Any] | None = None
        self._state_error: Exception | None = None
        self._state_ready = threading.Event()

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_ready = threading.Event()

        self._is_connected = False

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        keys = [f"{name}.pos" for name in JOINT_NAMES]
        keys += [f"gyro.{axis}" for axis in "xyz"]
        keys += [f"gravity.{axis}" for axis in "xyz"]
        keys += ["odom.x", "odom.y", "odom.yaw"]
        keys += [f"{name}.applied" for name in TWIST_ACTIONS]
        return dict.fromkeys(keys, float)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        if not self.config.use_camera:
            return {}
        return {CAMERA: (self.config.camera_height, self.config.camera_width, 3)}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(TWIST_ACTIONS + HEAD_ACTIONS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # Calibration lives on the robot.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        with self._state_lock:
            self._latest_state = None
            self._state_error = None
            self._state_ready.clear()
        with self._frame_lock:
            self._latest_frame = None
            self._frame_ready.clear()
        self._stop_event = threading.Event()
        try:
            try:
                self._cmd.open()
                self._state_rpc.open()
            except OSError as e:
                raise DeviceNotConnectedError(
                    f"Could not reach robotd at {self.config.robotd_address}. Forward its socket first: "
                    f"ssh -N -L {self.config.robotd_address}:/run/robotd.sock microduck@{self.config.remote_ip}"
                ) from e

            hello = self._cmd.request("hello", {"api_version": API_VERSION})
            if hello and hello.get("api_version") != API_VERSION:
                logger.warning(
                    f"robotd speaks API v{hello.get('api_version')}, this client was written for v{API_VERSION}."
                )

            subscribed = self._state_rpc.request("robot.subscribe", {"hz": self.config.state_hz})
            if subscribed and not subscribed.get("accepted", True):
                raise DeviceNotConnectedError(f"robotd refused the state subscription: {subscribed}")

            self._start_thread(self._state_loop, "microduck-state")
            if self.config.use_camera:
                self._start_thread(self._camera_loop, "microduck-camera")

            if not self._state_ready.wait(timeout=self.config.rpc_timeout_s):
                raise DeviceNotConnectedError("No robot.state received from robotd.")
            self._check_state_stream()
            if self.config.use_camera and not self._frame_ready.wait(
                timeout=self.config.camera_timeout_s * 3
            ):
                logger.warning(
                    f"No camera frame from http://{self.config.remote_ip}:{self.config.camera_port}/frame yet; "
                    "recording black frames until one arrives."
                )
            self._check_state_stream()

            if self.config.enable_on_connect:
                reply = self._cmd.request("robot.enable", {"on": True})
                if reply and not reply.get("accepted", True):
                    logger.warning(f"robot.enable was refused: {reply.get('reason')}")
            self._check_state_stream()
            self._is_connected = True
        except BaseException:
            self._shutdown_threads()
            self._is_connected = False
            raise
        logger.info(f"{self} connected.")

    def _check_state_stream(self) -> dict[str, Any]:
        """Return a state snapshot or raise if the stream failed."""
        with self._state_lock:
            error = self._state_error
            if error is not None:
                raise DeviceNotConnectedError(
                    f"robot.state stream stopped: {error}. Disconnect and reconnect {self}."
                ) from error
            return self._latest_state or {}

    def _start_thread(self, target, name: str) -> None:
        thread = threading.Thread(target=target, args=(self._stop_event,), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _state_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                message = self._state_rpc.read_message()
            except (OSError, ValueError, ConnectionError) as e:
                with self._state_lock:
                    if stop_event.is_set():
                        return
                    self._state_error = e
                    self._latest_state = None
                    self._state_ready.set()
                logger.error(f"robot.state stream stopped: {e}")
                return
            if message.get("method") != "robot.state":
                continue
            with self._state_lock:
                if stop_event.is_set():
                    return
                self._latest_state = message.get("params")
                self._state_ready.set()

    def _camera_loop(self, stop_event: threading.Event) -> None:
        url = f"http://{self.config.remote_ip}:{self.config.camera_port}/frame"
        size = (self.config.camera_width, self.config.camera_height)
        with requests.Session() as session:
            while not stop_event.is_set():
                try:
                    response = session.get(url, timeout=self.config.camera_timeout_s)
                    response.raise_for_status()
                except requests.RequestException as e:
                    logger.debug(f"Camera snapshot failed: {e}")
                    stop_event.wait(0.2)
                    continue
                frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                with self._frame_lock:
                    if stop_event.is_set():
                        return
                    self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._frame_ready.set()

    def _state_to_observation(self, state: dict[str, Any]) -> RobotObservation:
        obs: RobotObservation = {}
        joints = state.get("joints") or [0.0] * len(JOINT_NAMES)
        for name, value in zip(JOINT_NAMES, joints, strict=False):
            obs[f"{name}.pos"] = float(value)

        gyro = (state.get("imu") or {}).get("gyro") or [0.0, 0.0, 0.0]
        gravity = (state.get("safety") or {}).get("gravity") or [0.0, 0.0, 0.0]
        for i, axis in enumerate("xyz"):
            obs[f"gyro.{axis}"] = float(gyro[i])
            obs[f"gravity.{axis}"] = float(gravity[i])

        odom = state.get("odom") or {}
        position = odom.get("position") or [0.0, 0.0, 0.0]
        obs["odom.x"] = float(position[0])
        obs["odom.y"] = float(position[1])
        obs["odom.yaw"] = float(odom.get("yaw", 0.0))

        applied = (state.get("move") or {}).get("applied") or [0.0, 0.0, 0.0]
        for name, value in zip(TWIST_ACTIONS, applied, strict=True):
            obs[f"{name}.applied"] = float(value)
        return obs

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._check_state_stream()
        obs = self._state_to_observation(state)

        if self.config.use_camera:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                frame = np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)
            obs[CAMERA] = frame
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._check_state_stream()
        cfg = self.config
        limits = {"vx": cfg.max_vx, "vy": cfg.max_vy, "vyaw": cfg.max_vyaw}
        sent: RobotAction = {}
        for name in TWIST_ACTIONS:
            sent[name] = float(np.clip(float(action.get(name, 0.0)), -limits[name], limits[name]))
        for key in HEAD_ACTIONS:
            sent[key] = float(np.clip(float(action.get(key, 0.0)), -cfg.max_head, cfg.max_head))

        self._cmd.notify("robot.move", {name: sent[name] for name in TWIST_ACTIONS})
        self._cmd.notify("robot.head", {name: sent[f"{name}.pos"] for name in HEAD_JOINTS})
        return sent

    def get_health(self) -> dict[str, Any]:
        """robotd's ``robot.health`` reply: battery, motor temperatures, control loop, ..."""
        return self._cmd.request("robot.health")

    def load_policy(self, policy: str, slot: str = "walk") -> dict[str, Any]:
        """Loads an onboard ONNX policy into ``slot`` (``walk``, ``stand``, ``sitstand``, a skill, ...).

        ``policy`` is either an absolute path on the robot or a Hub repo (``user/repo[@revision][:file]``),
        in which case ``updaterd`` downloads it onto the robot first; that needs ``updaterd_address``.
        The swap happens at the home pose, so the reply only says the request was accepted.
        """
        if policy.startswith("/"):
            path = policy
        else:
            if self.config.updaterd_address is None:
                raise ValueError(
                    "Fetching a policy from the Hub goes through updaterd: forward /run/updaterd.sock and "
                    "set `updaterd_address`."
                )
            repo, _, file = policy.partition(":")
            repo, _, revision = repo.partition("@")
            params: dict[str, Any] = {"repo": repo}
            if revision:
                params["revision"] = revision
            if file:
                params["file"] = file
            updaterd = DuckRpc(self.config.updaterd_address, timeout_s=120.0)
            updaterd.open()
            try:
                fetched = updaterd.request("policy.fetch", params)
            finally:
                updaterd.close()
            path = fetched["path"]
            logger.info(f"Fetched {fetched.get('repo')}@{fetched.get('revision')} to {path}")
        return self._cmd.request("robot.loadPolicy", {"slot": slot, "path": path})

    def _shutdown_threads(self) -> None:
        self._stop_event.set()
        self._state_rpc.close()
        for thread in self._threads:
            thread.join(timeout=self.config.camera_timeout_s + 1.0)
        self._threads.clear()
        self._state_ready.clear()
        self._frame_ready.clear()
        self._cmd.close()

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self._cmd.request("robot.stop")
            if self.config.relax_on_disconnect:
                self._cmd.request("robot.relax")
        except (OSError, ConnectionError) as e:
            logger.warning(f"Could not stop {self} cleanly: {e}")
        finally:
            self._shutdown_threads()
            self._is_connected = False
        logger.info(f"{self} disconnected.")
