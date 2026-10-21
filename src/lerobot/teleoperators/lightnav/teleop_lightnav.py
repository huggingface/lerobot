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

"""Drive a mobile base with LightNav-0, a vision-language navigation model.

The model runs on a GPU host behind LightNav's WebSocket server, ``lightnav-serve``
(https://github.com/lightorigins/LightNav-0). This teleoperator is its robot-side client: it streams
one camera from the robot's observation and follows the predicted waypoints. It emits the planar base
actions of ``keyboard_base``, so ``--base_teleop.type=lightnav`` composes with a leader arm or holds
the arm still, like any other base input device.
"""

import base64
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from .config_lightnav import LightNavTeleopConfig

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

logger = logging.getLogger(__name__)

BASE_ACTION_KEYS = ("x.vel", "y.vel", "theta.vel")
# lightnav-serve accepts frames up to 64 MiB; a native-resolution JPEG is far below that.
_MAX_MESSAGE_SIZE = 64 * 1024 * 1024


@dataclass(frozen=True)
class LightNavPrediction:
    """One predicted chunk of cumulative ``[forward_m, lateral_m, yaw_rad]`` poses."""

    waypoints: np.ndarray
    # time.monotonic() when the frame it was predicted from was observed.
    frame_time: float
    stop: bool
    latency_ms: float


def waypoint_velocity(
    waypoints: np.ndarray, elapsed_s: float, dt_s: float
) -> tuple[float, float, float] | None:
    """Return the body velocity (m/s, m/s, rad/s) that follows a chunk ``elapsed_s`` after its frame.

    Rows are poses relative to the robot at the frame (+lateral = left, +yaw = counter-clockwise), one
    row per ``dt_s``. The segment active at ``elapsed_s`` is expressed in the frame of its start pose.
    Returns None once the chunk is exhausted.
    """
    index = max(0, int(elapsed_s // dt_s))
    if index >= len(waypoints):
        return None
    x0, y0, yaw0 = map(float, waypoints[index - 1]) if index > 0 else (0.0, 0.0, 0.0)
    x1, y1, yaw1 = map(float, waypoints[index])
    dx, dy = x1 - x0, y1 - y0
    forward = math.cos(yaw0) * dx + math.sin(yaw0) * dy
    lateral = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
    turn = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    return forward / dt_s, lateral / dt_s, turn / dt_s


def limit_velocity(
    x_vel: float, y_vel: float, theta_vel: float, max_linear: float, max_angular: float
) -> tuple[float, float, float]:
    """Scale the whole command down to the speed caps, so the robot keeps the planned curvature."""
    scale = 1.0
    linear = math.hypot(x_vel, y_vel)
    if linear > max_linear:
        scale = max_linear / linear
    if abs(theta_vel) > max_angular:
        scale = min(scale, max_angular / abs(theta_vel))
    return x_vel * scale, y_vel * scale, theta_vel * scale


def _encode_jpeg(image: np.ndarray, quality: int) -> str:
    ok, jpeg = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("Could not JPEG-encode the LightNav camera frame.")
    return base64.b64encode(jpeg.tobytes()).decode("ascii")


def _parse_waypoints(actions: Any) -> np.ndarray:
    rows = actions.get("actions") if isinstance(actions, dict) else None
    waypoints = np.asarray(rows, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0 or waypoints.shape[1] != 3:
        raise ValueError(f"LightNav returned malformed waypoints: {actions!r}")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("LightNav returned non-finite waypoints.")
    return waypoints


class LightNavTeleop(Teleoperator):
    """Follow LightNav-0 waypoints predicted from the robot's own camera.

    A background thread keeps one request in flight: it sends the newest frame delivered through
    ``on_observation`` and waits for its waypoints, so ``get_action`` never blocks on the network. The
    chunk is replayed open loop from the time its frame was observed and replaced by every new
    prediction. The base stops when the chunk runs out, when the model reports arrival (until
    ``reset_episode``), and when the connection fails.
    """

    config_class = LightNavTeleopConfig
    name = "lightnav"

    def __init__(self, config: LightNavTeleopConfig):
        super().__init__(config)
        self.config = config
        self._instruction = config.instruction
        self._ws: ClientConnection | None = None
        self._worker: threading.Thread | None = None
        self._closing = threading.Event()
        # Guards everything below, shared between the control loop and the worker thread.
        self._lock = threading.Condition()
        self._frame: tuple[float, np.ndarray] | None = None
        self._episode = 0
        self._reset_requested = False
        self._prediction: LightNavPrediction | None = None
        self._arrived = False
        self._seq = 0
        # The last camera object seen, to recognise a camera that repeats its latest picture.
        self._last_image: Any = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(BASE_ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def arrived(self) -> bool:
        """Whether the model reported that the goal is reached in the current episode."""
        with self._lock:
            return self._arrived

    @property
    def last_prediction(self) -> LightNavPrediction | None:
        with self._lock:
            return self._prediction

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        require_package("websockets", extra="lightnav")
        from websockets.sync.client import connect

        # Entered by hand because the connection outlives this call; disconnect() or the worker closes it.
        self._ws = connect(
            self.config.server_url, open_timeout=self.config.connect_timeout_s, max_size=_MAX_MESSAGE_SIZE
        ).__enter__()
        try:
            self._checked("login", {"clientId": f"lerobot-{self.id or self.name}"})
            self._checked("reset", {})
        except Exception:
            self._ws.close()
            self._ws = None
            raise
        with self._lock:
            # Discards a late reply from a previous connection's worker.
            self._episode += 1
            self._frame = None
            self._reset_requested = False
            self._prediction = None
            self._arrived = False
        self._last_image = None
        self._closing.clear()
        self._worker = threading.Thread(target=self._run, name="lightnav-client", daemon=True)
        self._worker.start()
        logger.info("Connected to LightNav at %s: %r", self.config.server_url, self._instruction)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def on_observation(self, observation: RobotObservation) -> None:
        """Queue the newest camera frame; an older frame that was not sent yet is dropped.

        A camera slower than the control loop often returns the same cached array until its next
        picture. That array is queued once: the server appends every frame it receives to the model's
        history, so repeats would compress the time the history spans.
        """
        key = self.config.camera_key
        if key not in observation:
            raise KeyError(
                f"LightNav camera {key!r} is not in the robot observation. "
                f"Set --base_teleop.camera_key to one of {sorted(observation)}."
            )
        if observation[key] is None or observation[key] is self._last_image:
            return
        self._last_image = observation[key]
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"LightNav needs an HxWx3 uint8 RGB image under {key!r}, got {image.dtype} {image.shape}."
            )
        with self._lock:
            self._frame = (time.monotonic(), image.copy())
            self._lock.notify()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._lock:
            prediction = None if self._arrived else self._prediction
        stopped = dict.fromkeys(BASE_ACTION_KEYS, 0.0)
        if prediction is None:
            return stopped
        velocity = waypoint_velocity(
            prediction.waypoints, time.monotonic() - prediction.frame_time, self.config.waypoint_dt_s
        )
        if velocity is None:
            return stopped
        x_vel, y_vel, theta_vel = velocity
        x_vel, y_vel, theta_vel = limit_velocity(
            x_vel,
            y_vel if self.config.holonomic else 0.0,
            math.degrees(theta_vel),
            self.config.max_linear_speed,
            self.config.max_angular_speed,
        )
        return dict(zip(BASE_ACTION_KEYS, (x_vel, y_vel, theta_vel), strict=True))

    def reset_episode(self, instruction: str | None = None) -> None:
        """Clear the server's frame history and resume after an arrival, optionally with a new goal."""
        if instruction is not None and not instruction.strip():
            raise ValueError("LightNav needs a non-empty navigation instruction.")
        with self._lock:
            if instruction is not None:
                self._instruction = instruction
            self._episode += 1
            self._reset_requested = True
            self._frame = None
            self._prediction = None
            self._arrived = False
            self._lock.notify()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self._closing.set()
        with self._lock:
            self._lock.notify()
        # Closing the socket also interrupts a pending reply; the worker may have closed it already.
        ws = self._ws
        if ws is not None:
            ws.close()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=self.config.response_timeout_s)

    def _has_work(self) -> bool:
        return (
            self._closing.is_set() or self._reset_requested or (self._frame is not None and not self._arrived)
        )

    def _run(self) -> None:
        try:
            while not self._closing.is_set():
                with self._lock:
                    if not self._lock.wait_for(self._has_work, timeout=0.1) or self._closing.is_set():
                        continue
                    if self._reset_requested:
                        self._reset_requested = False
                        frame = None
                    else:
                        frame, self._frame = self._frame, None
                    episode, instruction = self._episode, self._instruction
                if frame is None:
                    self._checked("reset", {})
                else:
                    self._predict(frame, episode, instruction)
        except Exception as exc:
            if not self._closing.is_set():
                logger.error("LightNav client stopped (%s): %s", type(exc).__name__, exc)
        finally:
            ws, self._ws = self._ws, None
            if ws is not None:
                ws.close()

    def _predict(self, frame: tuple[float, np.ndarray], episode: int, instruction: str) -> None:
        frame_time, image = frame
        self._seq += 1
        data = self._exchange(
            "next",
            {
                "seq": self._seq,
                "image": _encode_jpeg(image, self.config.jpeg_quality),
                "instruction": instruction,
            },
        )
        if data.get("seq") != self._seq:
            raise RuntimeError(f"LightNav answered request {data.get('seq')} instead of {self._seq}.")
        if data.get("rc") != 0:
            # The server keeps the session open; the current chunk keeps running until it is exhausted.
            logger.warning("LightNav could not predict (rc=%s): %s", data.get("rc"), data.get("msg"))
            return
        if "actions" not in data:
            return
        prediction = LightNavPrediction(
            waypoints=_parse_waypoints(data["actions"]),
            frame_time=frame_time,
            stop=bool(data.get("stop", False)),
            latency_ms=float(data.get("latency_ms", math.nan)),
        )
        with self._lock:
            if episode != self._episode:
                return
            self._prediction = prediction
            if prediction.stop and not self._arrived:
                self._arrived = True
                logger.info("LightNav reached the goal, the base stays stopped until reset_episode().")

    def _checked(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._exchange(action, payload)
        if data.get("rc") != 0:
            raise RuntimeError(f"LightNav {action} failed (rc={data.get('rc')}): {data.get('msg')}")
        return data

    def _exchange(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the ``data`` of its reply; one request is in flight at a time."""
        if self._ws is None:
            raise RuntimeError("LightNav is not connected.")
        self._ws.send(json.dumps({"action": action, "data": payload}))
        reply = json.loads(self._ws.recv(timeout=self.config.response_timeout_s))
        data = reply.get("data") if isinstance(reply, dict) else None
        if not isinstance(data, dict) or reply.get("action") != action:
            raise RuntimeError(f"Unexpected LightNav reply to {action}: {reply!r}")
        return data
