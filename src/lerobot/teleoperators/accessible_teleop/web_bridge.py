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

"""Local web server carrying operator input from a browser page to the teleoperator.

The page is served on loopback and talks over a WebSocket on the same port. The browser is
only an input surface: it reports raw channel readings and lets the operator edit their
mapping, while every decision that reaches a motor is taken in :mod:`.control`.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _websockets_available, require_package

from .config_accessible_teleop import (
    FACE_CHANNELS,
    JOYSTICK_CHANNELS,
    AccessibleTeleopConfig,
    ChannelCalibration,
    JointBinding,
    binding_from_dict,
    binding_to_dict,
)
from .control import InputFrame

if TYPE_CHECKING or _websockets_available:
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    from websockets.sync.server import Server, ServerConnection, serve
else:
    Headers = Response = Server = ServerConnection = serve = None

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"

# Fast enough that the page feels attached to the robot, slow enough that the readout does
# not dominate the socket when the control loop is running at 60 Hz.
STATE_BROADCAST_HZ = 20.0


class ControlBridge:
    """Serves the control page and holds the most recent operator input.

    Everything the page sends lands in this object under a lock; the teleoperator drains it
    from the control loop thread.
    """

    def __init__(
        self,
        config: AccessibleTeleopConfig,
        bindings: dict[str, JointBinding],
        calibrations: dict[str, ChannelCalibration],
        on_profile_change: Callable[[], None] | None = None,
    ):
        require_package("websockets", extra="accessible-teleop")
        self.config = config
        self.bindings = bindings
        self.calibrations = calibrations
        self._on_profile_change = on_profile_change

        self._lock = threading.Lock()
        # Input is tracked per page rather than globally: a second page left open on another
        # screen must not be able to overwrite the input of the page actually being driven.
        self._frames: dict[int, tuple[InputFrame, float | None]] = {}
        self._next_client_id = 0
        self._client_seen = threading.Event()

        self._server: Server | None = None
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {}

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._server is not None

    @property
    def bound_port(self) -> int:
        """Port the server actually listens on, which differs from the config when it is 0."""
        if self._server is None:
            return self.config.web_port
        return self._server.socket.getsockname()[1]

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.bound_port}/"

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def start(self) -> None:
        self._server = serve(
            self._handle_connection,
            self.config.host,
            self.config.web_port,
            process_request=self._process_request,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="accessible-teleop-web", daemon=True
        )
        self._thread.start()
        logger.info(f"Accessible teleop control page served at {self.url}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def wait_for_client(self, timeout_s: float) -> bool:
        """Block until the control page attaches, or the timeout expires."""
        return self._client_seen.wait(timeout_s)

    # ── operator input ───────────────────────────────────────────────────

    def read_frame(self) -> tuple[InputFrame, float | None]:
        """Return the input frame in charge of the robot, and its age in seconds.

        With several pages open, the one holding its clutch closed wins; among equals the
        freshest frame wins. Without that rule an idle second page would overwrite the
        driving page's input sixty times a second, and the clutch could never latch.

        The age is ``None`` when no frame has arrived yet. A stale frame is the caller's
        problem to act on: the bridge never invents input.
        """
        now = time.monotonic()
        with self._lock:
            stamped = [(frame, at) for frame, at in self._frames.values() if at is not None]
            if not stamped:
                return InputFrame(), None
            # A page that stopped reporting, because it was backgrounded or its tab froze,
            # must not keep the clutch it was holding when it went quiet.
            fresh = [entry for entry in stamped if now - entry[1] <= self.config.input_timeout_s]
            engaged = [entry for entry in fresh if entry[0].engaged]
            frame, at = max(engaged or fresh or stamped, key=lambda entry: entry[1])
            frame = InputFrame(
                channels=dict(frame.channels),
                keys=dict(frame.keys),
                engaged=frame.engaged,
                tracking=frame.tracking,
            )
        return frame, now - at

    def release_clutch(self) -> None:
        """Force the clutch open, e.g. because input went stale or the loop is shutting down."""
        with self._lock:
            for client_id, (frame, at) in self._frames.items():
                frame.engaged = False
                self._frames[client_id] = (frame, at)

    def publish_state(self, state: dict[str, Any]) -> None:
        """Hand the page the robot-facing state it should display."""
        with self._lock:
            self._state = state

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _process_request(self, connection: "ServerConnection", request: Any) -> "Response | None":
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            return None  # let websockets perform the upgrade
        if path == "/":
            return self._html_response()
        return Response(404, "Not Found", Headers({"Content-Length": "0"}), b"")

    def _html_response(self) -> "Response":
        body = (ASSETS_DIR / "index.html").read_text(encoding="utf-8")
        body = body.replace("__LEROBOT_BOOTSTRAP__", json.dumps(self._bootstrap()))
        payload = body.encode("utf-8")
        headers = Headers(
            {
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": str(len(payload)),
                "Cache-Control": "no-store",
            }
        )
        return Response(200, "OK", headers, payload)

    def _bootstrap(self) -> dict[str, Any]:
        with self._lock:
            bindings = {joint: binding_to_dict(b) for joint, b in self.bindings.items()}
            calibrations = {ch: asdict(c) for ch, c in self.calibrations.items()}
        return {
            "joints": list(self.config.joints),
            "jointLimits": {j: list(v) for j, v in self.config.joint_limits.items()},
            "faceChannels": list(FACE_CHANNELS) if self.config.face_tracking else [],
            "joystickChannels": list(JOYSTICK_CHANNELS),
            "bindings": bindings,
            "calibrations": calibrations,
            "faceTracking": self.config.face_tracking,
            "mediapipeBaseUrl": self.config.mediapipe_base_url,
            "faceLandmarkerUrl": self.config.face_landmarker_url,
            "robotId": self.config.id or "unnamed",
        }

    # ── WebSocket ────────────────────────────────────────────────────────

    def _handle_connection(self, connection: "ServerConnection") -> None:
        with self._lock:
            client_id = self._next_client_id
            self._next_client_id += 1
            self._frames[client_id] = (InputFrame(), None)
            client_count = len(self._frames)
        self._client_seen.set()
        logger.info("Control page connected")
        if client_count > 1:
            logger.warning(
                f"{client_count} control pages are open; only the one holding its clutch "
                "closed drives the robot."
            )

        state_period = 1.0 / STATE_BROADCAST_HZ
        next_state = 0.0
        try:
            while True:
                try:
                    message = connection.recv(timeout=state_period)
                except TimeoutError:
                    message = None
                if message is not None:
                    self._handle_message(client_id, message)

                now = time.monotonic()
                if now >= next_state:
                    next_state = now + state_period
                    with self._lock:
                        state = dict(self._state)
                        state["clients"] = len(self._frames)
                    if state:
                        connection.send(json.dumps({"type": "state", **state}))
        except Exception as exc:  # noqa: BLE001 - a dropped page must not kill the robot loop
            logger.debug(f"Control page connection ended: {exc}")
        finally:
            # A page that goes away takes its input, and any clutch it was holding, with it.
            with self._lock:
                self._frames.pop(client_id, None)
            logger.info("Control page disconnected")

    def _handle_message(self, client_id: int, message: str | bytes) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            logger.warning("Discarding malformed message from control page")
            return

        kind = payload.get("type")
        if kind == "input":
            self._apply_input(client_id, payload)
        elif kind == "profile":
            self._apply_profile(payload)
        elif kind == "stop":
            self.release_clutch()
        else:
            logger.debug(f"Ignoring unknown message type from control page: {kind!r}")

    def _apply_input(self, client_id: int, payload: dict[str, Any]) -> None:
        channels = payload.get("channels") or {}
        keys = payload.get("keys") or {}
        frame = InputFrame(
            channels={str(k): float(v) for k, v in channels.items() if _is_number(v)},
            keys={str(k): bool(v) for k, v in keys.items()},
            engaged=bool(payload.get("engaged", False)),
            tracking=bool(payload.get("tracking", False)),
        )
        with self._lock:
            self._frames[client_id] = (frame, time.monotonic())

    def _apply_profile(self, payload: dict[str, Any]) -> None:
        bindings = payload.get("bindings") or {}
        calibrations = payload.get("calibrations") or {}
        with self._lock:
            for joint, raw in bindings.items():
                if joint in self.bindings:
                    self.bindings[joint] = binding_from_dict(raw)
            for channel, raw in calibrations.items():
                self.calibrations[channel] = ChannelCalibration(
                    neutral=float(raw.get("neutral", 0.0)),
                    negative_range=float(raw.get("negative_range", 1.0)),
                    positive_range=float(raw.get("positive_range", 1.0)),
                    ready=bool(raw.get("ready", False)),
                )
        logger.info("Applied updated control profile from the page")
        if self._on_profile_change is not None:
            self._on_profile_change()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
