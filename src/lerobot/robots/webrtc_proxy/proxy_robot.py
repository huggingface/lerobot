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

"""Cloud-side ``WebRTCProxyRobot`` — a fake robot that proxies a real one on a robot.

LeRobot's record/teleop/policy code calls ``get_observation`` / ``send_action`` /
``observation_features`` synchronously and assumes they are local and instant. This
class honours that synchronous contract while the work happens over WebRTC: it owns
a background asyncio loop running an :class:`_ProxyEndpoint` (the *answerer*).

- ``get_observation`` reads the thread-safe :class:`AlignmentBuffer` (no loop hop) and
  assembles the LeRobot obs dict by *capture* timestamp (handoff challenge A).
- ``send_action`` marshals an :class:`ActionMsg` onto the loop's action DataChannel.
- the robot-side watchdog (not here) handles disconnect safety (handoff challenge C).

M1: ``signaling_url`` unset => loopback mode, which also spins up a synthetic
:class:`CaptureAgent` *in the same loop* so the whole link is self-contained and
testable on one machine. Real mode (M3) replaces loopback signaling with the K8s
WebSocket signaler and drops the in-process capture agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from . import tiling
from .alignment import AlignmentBuffer
from .capture_agent import _fit_frame
from .configuration_webrtc_proxy import WebRTCProxyRobotConfig
from .control import ControlClient
from .protocol import CH_ACTION, CH_CONTROL, CH_STATE, ActionMsg, StateMsg
from .signaling import Signaling, WebSocketSignaling
from .transport import Transport, make_transport

logger = logging.getLogger(__name__)


class CameraLayoutMismatchError(ValueError):
    """The robot's tiled video frame doesn't fit this end's camera specs.

    Cameras are tiled into one track and sliced back purely by the (name-sorted)
    specs — there is no per-frame metadata — so the two ends must declare the SAME
    camera set (same names AND per-camera WxH). When they disagree the slices are
    wrong, and a too-short frame would otherwise crash deep in ``cv2.resize`` on an
    empty tile. We raise this instead, with an actionable message.
    """


class _ProxyEndpoint:
    """Cloud answerer: receives state + video, sends actions, drives the control plane.

    Transport-agnostic — talks to a :class:`Transport` (default aiortc).
    """

    def __init__(
        self,
        buffer: AlignmentBuffer,
        cam_name: str,
        ice_servers: list[str] | None = None,
        transport_backend: str = "aiortc",
        livekit_url: str | None = None,
        livekit_token: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.buffer = buffer
        self.cam_name = cam_name
        self._action_seq = 0
        # Closed-loop feedback reported by the robot on the state stream (telemetry).
        self.last_applied_seq = -1
        self.last_applied_t = 0.0
        self._control = ControlClient()
        self._transport = transport or make_transport(
            transport_backend,
            role="subscriber",
            channels={CH_STATE: False, CH_ACTION: False, CH_CONTROL: True},  # reliability set by publisher
            ice_servers=ice_servers,
            livekit_url=livekit_url,
            livekit_token=livekit_token,
        )
        self.connected = self._transport.connected
        self._transport.channel(CH_STATE).on_message(self._on_state)
        self._control.attach(self._transport.channel(CH_CONTROL))
        self._transport.set_frame_handler(self._on_frame)

    def _on_state(self, raw: str) -> None:
        try:
            msg = StateMsg.from_json(raw)
        except Exception:
            logger.exception("proxy: bad state message")
            return
        self.buffer.add_state(msg.seq, msg.t, msg.joints)
        if msg.applied_seq > self.last_applied_seq:
            self.last_applied_seq = msg.applied_seq
            self.last_applied_t = msg.applied_t

    def _on_frame(self, seq: int, img: np.ndarray) -> None:
        # seq was recovered from the frame pts by the transport. A dropped frame just
        # leaves a gap — no cascade. (Relative to the first received frame; DESIGN §5.1.)
        self.buffer.add_frame(seq, img)

    async def run(self, signaling: Signaling) -> None:
        await self._transport.open(signaling)

    async def send_action(self, goal: dict[str, float], obs_seq: int = -1) -> dict[str, float]:
        self._action_seq += 1
        self._transport.channel(CH_ACTION).send(
            ActionMsg(t=time.monotonic(), seq=self._action_seq, goal=goal, obs_seq=obs_seq).to_json()
        )
        return goal

    async def control_call(self, method: str, params: dict | None = None, timeout: float = 10.0):
        return await self._control.call(method, params, timeout)

    async def close(self) -> None:
        await self._transport.close()


class _EventLoopThread:
    """Owns an asyncio loop in a daemon thread; bridges sync calls to coroutines."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="webrtc-proxy-loop", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)


class WebRTCProxyRobot(Robot):
    """Cloud-side proxy presenting a real robot-tethered robot as a local LeRobot Robot."""

    config_class = WebRTCProxyRobotConfig
    name = "webrtc_proxy"

    def __init__(self, config: WebRTCProxyRobotConfig):
        super().__init__(config)
        self.config = config
        if not config.cameras:
            raise ValueError("WebRTCProxyRobot needs at least one camera")
        # Cameras are tiled into one video track on the robot and sliced back here; both ends
        # derive the same layout from the (name-sorted) specs. See tiling.py.
        self._cam_specs = {name: (spec.height, spec.width) for name, spec in config.cameras.items()}
        self._specs = tiling.ordered_specs(self._cam_specs)
        # The first camera (name order) backs the single-camera set_camera_plan hint.
        self.cam_name, self.cam_spec = sorted(config.cameras.items())[0]
        self.motors = list(config.motors)

        self._buffer = AlignmentBuffer()
        self._loop: _EventLoopThread | None = None
        self._endpoint: _ProxyEndpoint | None = None
        self._ws_sig: WebSocketSignaling | None = None
        self._last_obs: RobotObservation | None = None  # last coherent obs (held on camera lag)
        self._last_obs_seq = -1  # seq of the most recent obs returned (action provenance)
        self._connected = False
        self._frame_checked = False  # one-time camera-layout sanity check (see get_observation)

    # ----- schema (callable whether connected or not) ----------------------
    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.motors}

    @property
    def observation_features(self) -> dict:
        cams = {name: (spec.height, spec.width, 3) for name, spec in self.config.cameras.items()}
        return {**self._motors_ft, **cams}

    @property
    def action_features(self) -> dict:
        return dict(self._motors_ft)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ----- no-op hardware hooks (calibration lives on the robot, M2) ----------
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ----- lifecycle -------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise RuntimeError("WebRTCProxyRobot already connected")
        # aiortc reaches the daemon over our WS signaling relay; livekit signals itself.
        if self.config.transport_backend == "aiortc" and (
            not self.config.signaling_url or not self.config.signaling_url.startswith("ws")
        ):
            raise ValueError("WebRTCProxyRobot (aiortc) needs a WebSocket signaling_url (ws://host:port/ws)")

        self._loop = _EventLoopThread()
        self._loop.start()

        # CRITICAL: every aiortc object (RTCPeerConnection), asyncio.Event and
        # asyncio.Queue must be constructed *on the loop thread* so it binds to the
        # right running loop. Building them in the caller thread silently deadlocks.
        async def _bringup() -> None:
            self._endpoint = _ProxyEndpoint(
                self._buffer,
                self.cam_name,
                ice_servers=self.config.ice_servers,
                transport_backend=self.config.transport_backend,
                livekit_url=self.config.livekit_url,
                livekit_token=self.config.livekit_token,
            )
            # aiortc reaches the daemon over our WS signaling relay; livekit signals itself
            # (the transport ignores the signaling arg).
            if self.config.transport_backend == "aiortc":
                self._ws_sig = WebSocketSignaling(
                    self.config.signaling_url,
                    self.config.session_id,
                    role="controller",
                    token=self.config.signaling_token,
                )
            # Staged so a connect timeout points at the stuck phase (signaling vs media).
            logger.info(
                "connecting: signaling %s (session=%s)…",
                self.config.signaling_url,
                self.config.session_id,
            )
            await self._endpoint.run(self._ws_sig)
            logger.info("connecting: signaling done; waiting for media (ICE) to connect…")
            await self._endpoint.connected.wait()
            logger.info("connecting: media connected")
            # Push our desired obs size so the robot resizes/encodes to it (bandwidth).
            # Correctness doesn't depend on this — get_observation re-fits to the spec.
            with contextlib.suppress(Exception):
                await self._endpoint.control_call(
                    "set_camera_plan",
                    {"width": self.cam_spec.width, "height": self.cam_spec.height, "fps": self.cam_spec.fps},
                    timeout=5.0,
                )

        try:
            self._loop.run(_bringup(), timeout=self.config.connect_timeout_s)
        except TimeoutError as e:
            raise TimeoutError(
                f"WebRTCProxyRobot connect timed out after {self.config.connect_timeout_s}s. Check: "
                "(1) the signaling relay is reachable at the signaling_url; "
                "(2) the robot daemon is connected to the SAME relay with --session "
                f"'{self.config.session_id}' and a matching --auth-token; "
                "(3) for a cross-NAT/public link, STUN/TURN is configured on the relay — host-only ICE "
                "(ice_servers=[]) can't traverse the internet. See the 'connecting:' logs above for the "
                "phase it stalled in (signaling vs media)."
            ) from e
        self._wait_first_obs(self.config.connect_timeout_s)
        self._connected = True
        logger.info("WebRTCProxyRobot connected (%s)", self.config.signaling_url)

    def _wait_first_obs(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._buffer.assemble() is not None:
                return  # a state+frame pair (same seq) is available
            time.sleep(0.02)
        raise TimeoutError("WebRTCProxyRobot: no aligned observation within connect_timeout_s")

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("WebRTCProxyRobot not connected")
        # The freshest seq present on both the state and frame sides — an exact pair from
        # one capture instant. A dropped frame/state just means that seq is skipped.
        aligned = self._buffer.assemble()
        if aligned is not None:
            if not self._frame_checked:
                self._frame_checked = True
                self._check_frame_layout(aligned.frame)
            self._last_obs_seq = aligned.seq  # actions sent next derive from this obs
            obs: RobotObservation = dict(aligned.joints)
            # The robot tiled all cameras into one frame; slice it back per camera (same
            # name-sorted layout) and enforce each declared obs shape.
            for name, tile in tiling.untile(aligned.frame, self._specs).items():
                spec = self.config.cameras[name]
                obs[name] = _fit_frame(tile, spec.height, spec.width)
            self._last_obs = obs
            return dict(obs)
        # No complete pair yet (camera lag/stall): hold the last good obs rather than
        # fabricate one. Raise only if we never had one.
        if self._last_obs is not None:
            logger.warning("no state+frame pair yet; holding last obs")
            return dict(self._last_obs)
        raise RuntimeError("no observation available yet")

    def _check_frame_layout(self, frame: np.ndarray) -> None:
        """Verify the robot's first tiled frame matches this end's camera specs.

        Both ends slice the single video track by the same name-sorted specs, so a
        size disagreement means the two ``--cameras`` sets differ. A frame shorter
        than the stacked specs would leave empty tiles and crash ``cv2.resize``;
        raise :class:`CameraLayoutMismatchError` with a fix-it message instead. A taller/
        wider frame still de-tiles (each tile is re-fit), so warn rather than fail.
        """
        exp_h, exp_w = tiling.tiled_size(self._specs)
        fh, fw = frame.shape[:2]
        if (fh, fw) == (exp_h, exp_w):
            return
        names = [n for n, _, _ in self._specs]
        per_cam = ", ".join(f"{n}:{w}x{h}" for n, h, w in self._specs)
        if fh < exp_h or fw < exp_w:
            raise CameraLayoutMismatchError(
                f"camera set mismatch: this end is configured for {len(names)} camera(s) "
                f"[{per_cam}] — expecting a {exp_w}x{exp_h} tiled frame — but the robot daemon "
                f"is streaming a {fw}x{fh} frame (too small to hold them). Both ends must use "
                f"the SAME --cameras: identical names AND per-camera WxH. Update --cameras on "
                f"the robot daemon or here so they match."
            )
        logger.warning(
            "camera frame is %dx%d but this end's --cameras [%s] expect %dx%d; "
            "de-tiling may be misaligned — make both ends use the same --cameras.",
            fw,
            fh,
            per_cam,
            exp_w,
            exp_h,
        )

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected or self._endpoint is None or self._loop is None:
            raise RuntimeError("WebRTCProxyRobot not connected")
        goal = {k: float(v) for k, v in action.items() if k.endswith(".pos")}
        return self._loop.run(self._endpoint.send_action(goal, self._last_obs_seq), timeout=2.0)

    # ----- control plane: cloud-driven device onboarding (M3) ---------------
    # These reach the *robot's* OS over the control channel; port/camera IDs never
    # live in the cloud config. find_port is two-step because the human unplugs
    # the bus on the robot between the calls (the cloud cannot share that stdin).
    def _control(self, method: str, params: dict | None = None, timeout: float = 10.0):
        if not self._connected or self._endpoint is None or self._loop is None:
            raise RuntimeError("WebRTCProxyRobot not connected")
        return self._loop.run(self._endpoint.control_call(method, params, timeout), timeout=timeout + 1.0)

    def list_ports(self) -> list[str]:
        """Serial ports currently visible on the robot."""
        return self._control("list_ports")["ports"]

    def list_cameras(self) -> list[dict]:
        """Cameras on the robot, each with a stable id (opencv index_or_path / realsense serial)."""
        return self._control("list_cameras")["cameras"]

    def find_port_begin(self) -> list[str]:
        """Step 1/2: snapshot ports, then prompt the user (robot-side) to unplug the bus."""
        return self._control("find_port_begin")["ports"]

    def find_port_result(self) -> str:
        """Step 2/2 (after the user unplugged): the port that disappeared = the bus."""
        return self._control("find_port_result")["port"]

    def grab_camera_preview(self, cam_id, width: int = 320, height: int = 240) -> np.ndarray:
        """Grab one frame from camera ``cam_id`` on the robot (RGB uint8 ndarray).

        For onboarding previews — pick which physical camera is "front"/"wrist" before
        pinning the stream. Opens the camera on the robot, so it fails if that camera is
        already being streamed; preview the others first, then start streaming.
        """
        from .control import _decode_jpeg_b64

        res = self._control("grab_camera", {"id": cam_id, "width": width, "height": height}, timeout=15.0)
        return _decode_jpeg_b64(res["jpeg_b64"])

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._loop is not None:
            if self._endpoint is not None:
                try:
                    self._loop.run(self._endpoint.close(), timeout=2.0)
                except Exception:
                    logger.exception("error closing proxy endpoint")
            if self._ws_sig is not None:
                try:
                    self._loop.run(self._ws_sig.close(), timeout=2.0)
                except Exception:
                    logger.exception("error closing signaling")
            self._loop.stop()
        self._connected = False
