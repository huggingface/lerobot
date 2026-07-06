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

"""robot-side capture agent (the *offerer*).

Without a real ``robot`` it produces a *synthetic* source (no serial bus / camera).
It runs the capture clock, pushes proprioceptive state over the state DataChannel,
streams the camera over the media track (each frame's capture seq carried in its
pts), receives actions, answers control-plane RPCs, and runs the P0 safety watchdog.

The agent owns the WebRTC offer: it creates the state/action/control DataChannels and
adds the video track, then hands its SDP to the signaling channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import tiling
from .control import ControlServer, DeviceInventory, SyntheticInventory
from .protocol import CH_ACTION, CH_CONTROL, CH_STATE, ActionMsg, StateMsg
from .signaling import Signaling, SignalingClosedError
from .transport import Transport, make_transport

logger = logging.getLogger(__name__)


def _fit_frame(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Coerce a camera frame to the contiguous ``(height, width, 3)`` uint8 RGB the
    media track requires (the cloud declared this shape in ``observation_features``)."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[0] != height or img.shape[1] != width:
        import cv2

        img = cv2.resize(img, (width, height))
    return np.ascontiguousarray(img)


class CaptureAgent:
    """robot-side endpoint of the proxy link: publishes state + camera, applies actions.

    Transport-agnostic: it talks to a :class:`Transport` (default aiortc). Swap the
    transport for another backend (e.g. LiveKit) without touching this logic.
    """

    def __init__(
        self,
        signaling: Signaling | None,  # None for livekit (it does its own signaling)
        motors: list[str],
        cam_name: str | None = None,  # single-camera (back-compat); else use `cameras`
        cam_height: int | None = None,
        cam_width: int | None = None,
        capture_fps: int = 30,
        action_timeout_s: float = 0.5,
        on_safe_stop: Callable[[], None] | None = None,
        inventory: DeviceInventory | None = None,
        ice_servers: list[str] | None = None,
        camera=None,  # an opened lerobot Camera (read_latest); None => synthetic frames
        cameras: dict[str, tuple[int, int]] | None = None,  # multi-cam: name -> (height, width)
        cameras_src: dict | None = None,  # multi-cam: name -> opened Camera (read_latest)
        robot=None,  # a connected lerobot Robot (so_follower) — drives joints+action+torque (M2)
        reliable_state: bool = False,  # True for record (no lost obs); False for teleop/eval (fresh)
        reliable_action: bool = False,
        transport_backend: str = "aiortc",  # "aiortc" (default) | "livekit"
        livekit_url: str | None = None,
        livekit_token: str | None = None,
        transport: Transport | None = None,  # explicit transport overrides the backend
    ) -> None:
        self.signaling = signaling
        self.motors = list(motors)
        # Single camera keeps cam_name/cam_h/cam_w (dynamic via set_camera_plan); multi
        # camera uses a fixed `cameras` map. _capture_sample tiles whichever applies, and a
        # single camera tiles to the identity frame (so single-cam stays byte-identical).
        self._multi_cameras = dict(cameras) if cameras else None
        self.cam_name = cam_name
        self.cam_h = cam_height
        self.cam_w = cam_width
        self._cams_src = dict(cameras_src) if cameras_src else ({cam_name: camera} if camera else {})
        self._last_frames: dict[str, np.ndarray] = {}  # per-camera last good frame
        self.period = 1.0 / capture_fps
        self.action_timeout_s = action_timeout_s
        self._on_safe_stop = on_safe_stop

        # The transport offers + sends video, exposes the data channels. The publisher
        # (robot) sets channel reliability; control is always reliable.
        self._transport = transport or make_transport(
            transport_backend,
            role="publisher",
            channels={CH_STATE: reliable_state, CH_ACTION: reliable_action, CH_CONTROL: True},
            ice_servers=ice_servers,
            livekit_url=livekit_url,
            livekit_token=livekit_token,
        )
        self.closed = self._transport.closed  # set when the link drops
        self._transport.channel(CH_ACTION).on_message(self._on_action)
        self._control = ControlServer(
            inventory if inventory is not None else SyntheticInventory(),
            on_camera_plan=self._apply_camera_plan,
        )
        self._control.attach(self._transport.channel(CH_CONTROL))
        self._robot = robot
        # All serial-bus access (read joints, send action, toggle torque) goes through
        # ONE worker thread so the public-net event loop never blocks on serial and the
        # bus is never touched concurrently. None when there is no real robot.
        self._io: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="webrtc-robot-io")
            if robot is not None
            else None
        )
        self._seq = 0
        self._action_seq = 0  # last action seq applied (for telemetry)
        self._last_obs_seq_seen = -1  # provenance of the last action (which obs it came from)
        self._last_goal: dict[str, float] = {f"{m}.pos": 0.0 for m in self.motors}
        self._last_action_t = time.monotonic()
        self._safed = False
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ----- lifecycle -------------------------------------------------------
    async def run(self) -> None:
        """Establish the transport (offer) and start the capture/watchdog loops.

        Returns once the loops are running; use :pymeth:`wait_closed` to block until
        the link drops.
        """
        await self._transport.open(self.signaling)

        if self._robot is not None and self._io is not None:
            # New session: re-enable torque (a previous session's safe-stop may have cut it).
            self._io.submit(self._robot_enable_torque)

        self._tasks = [
            asyncio.ensure_future(self._capture_loop()),
            asyncio.ensure_future(self._watchdog_loop()),
        ]
        # Recycle fast: keep watching the signaling channel so the relay's "bye" (controller
        # left) ends the session in sub-second time, instead of waiting ~10-30s for WebRTC's
        # own ICE/DTLS disconnect detection. Only for aiortc (livekit has its own signaling).
        if self.signaling is not None:
            self._tasks.append(asyncio.ensure_future(self._watch_signaling()))
        logger.info(
            "CaptureAgent connected; streaming %d motors + camera %r", len(self.motors), self.cam_name
        )

    async def _watch_signaling(self) -> None:
        """End the session promptly when the relay reports the controller left ("bye").

        After the offer/answer exchange the signaling socket is otherwise idle; reading it
        lets us notice the peer's departure immediately and recycle for the next session,
        rather than waiting for the media link's slow timeout.
        """
        try:
            while not self._stop.is_set():
                await self.signaling.recv()  # blocks; raises SignalingClosedError on bye / close
        except SignalingClosedError:
            logger.info("signaling: controller left (bye) -> ending session for fast recycle")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("signaling watcher error")
        finally:
            self.closed.set()  # unblock wait_closed() -> daemon loops to the next session

    async def wait_closed(self) -> None:
        """Block until the WebRTC link to the controller drops."""
        await self.closed.wait()

    def force_safe_stop(self) -> None:
        """Idempotently engage the safe state (used when a session ends)."""
        if not self._safed:
            self._safed = True
            self._safe_stop()

    async def close(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await self._transport.close()
        if self.signaling is not None:
            await self.signaling.close()
        if self._io is not None:
            # Let a pending safe-stop (disable_torque) finish, then stop the io thread.
            self._io.shutdown(wait=True)

    # ----- capture / actuation -------------------------------------------
    def _specs(self) -> list[tuple[str, int, int]]:
        """Ordered (name, h, w) per camera. Single camera tracks the (plan-mutable)
        cam_name/cam_h/cam_w; multi camera is the fixed ``cameras`` map."""
        if self._multi_cameras is not None:
            return tiling.ordered_specs(self._multi_cameras)
        return [(self.cam_name, self.cam_h, self.cam_w)]

    def _capture_sample(self, t: float, seq: int) -> tuple[dict[str, float], np.ndarray]:
        """One sample: joints + a (possibly multi-camera, tiled) frame.

        With a real ``robot``, joints and every camera come from a single
        ``robot.get_observation()`` (so they share a capture instant). Else: each camera is
        a real one if attached (``cameras_src``), otherwise a synthetic per-camera coloured
        frame; joints hold the last commanded pose. Frames are tiled into one (single
        camera => identity), preserving the one-frame-per-seq pipeline.
        """
        specs = self._specs()
        obs = self._robot.get_observation() if self._robot is not None else None
        if obs is not None:
            joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        else:
            joints = dict(self._last_goal)  # synthetic arm: hold last commanded pose

        frames: dict[str, np.ndarray] = {}
        for idx, (name, h, w) in enumerate(specs):
            frames[name] = self._camera_frame(name, h, w, seq, idx, obs)
        return joints, tiling.tile(frames, specs)

    def _camera_frame(self, name, h, w, seq, idx, obs) -> np.ndarray:  # noqa: ANN001
        """One camera's frame at ``(h, w)``: from the robot obs, a real camera, or synthetic."""
        if obs is not None:
            frame = obs.get(name)
            if frame is not None:
                self._last_frames[name] = _fit_frame(frame, h, w)
        else:
            src = self._cams_src.get(name)
            if src is not None:
                # camera warming up / stale: fall through to last good (or synthetic)
                with contextlib.suppress(Exception):
                    self._last_frames[name] = _fit_frame(src.read_latest(max_age_ms=1000), h, w)
            elif name not in self._last_frames:
                # Synthetic, distinct per camera (idx) so each tile is identifiable.
                img = np.empty((h, w, 3), dtype=np.uint8)
                img[:] = ((seq + idx * 64) % 256, (seq * 5) % 256, (idx * 80) % 256)
                return img
        got = self._last_frames.get(name)
        return got if got is not None else np.zeros((h, w, 3), dtype=np.uint8)

    def _apply_action(self, goal: dict[str, float]) -> dict[str, float]:
        """Drive the arm (runs on the io thread). Returns the action actually sent."""
        if self._robot is not None:
            try:
                return self._robot.send_action(goal)
            except Exception:
                logger.exception("CaptureAgent: robot.send_action failed")
                return goal
        self._last_goal = dict(goal)
        return self._last_goal

    def _robot_enable_torque(self) -> None:
        try:
            self._robot.bus.enable_torque()
        except Exception:
            logger.exception("CaptureAgent: enable_torque failed")

    def _robot_disable_torque(self) -> None:
        try:
            self._robot.bus.disable_torque()
            logger.warning("CaptureAgent: torque disabled (safe stop)")
        except Exception:
            logger.exception("CaptureAgent: disable_torque failed")

    def _apply_camera_plan(self, plan: dict) -> None:
        """Cloud told us its desired obs size — encode/resize frames to it (bandwidth)."""
        w, h = plan.get("width"), plan.get("height")
        if w and h and (w != self.cam_w or h != self.cam_h):
            logger.info("camera plan: obs size %dx%d -> %dx%d", self.cam_w, self.cam_h, w, h)
            self.cam_w, self.cam_h = int(w), int(h)

    def _safe_stop(self) -> None:
        """P0: watchdog fired (actions stopped). Cut torque so the arm goes limp."""
        logger.warning("WATCHDOG: no action for %.0fms -> SAFE STOP", self.action_timeout_s * 1e3)
        if self._robot is not None and self._io is not None:
            self._io.submit(self._robot_disable_torque)  # never touch the bus off the io thread
        if self._on_safe_stop is not None:
            self._on_safe_stop()

    # ----- loops -----------------------------------------------------------
    async def _capture_loop(self) -> None:
        loop = asyncio.get_event_loop()
        next_t = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            seq = self._seq
            self._seq += 1
            # Real serial reads run off-loop so public-net timing never blocks on the bus.
            if self._io is not None:
                joints, img = await loop.run_in_executor(self._io, self._capture_sample, t, seq)
            else:
                joints, img = self._capture_sample(t, seq)

            # Piggyback the last applied action (seq + time) so the cloud can confirm
            # landing and measure round-trip without an extra channel.
            self._transport.channel(CH_STATE).send(
                StateMsg(
                    t=t,
                    seq=seq,
                    joints=joints,
                    applied_seq=self._action_seq,
                    applied_t=self._last_action_t,
                ).to_json()
            )
            # The frame carries its seq in its pts (set inside the transport).
            self._transport.send_frame(seq, img)

            next_t += self.period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    def _on_action(self, raw: str) -> None:
        try:
            msg = ActionMsg.from_json(raw)
        except Exception:
            logger.exception("CaptureAgent: bad action message")
            return
        self._last_action_t = time.monotonic()  # sync: keeps the watchdog honest
        self._action_seq = msg.seq
        self._last_obs_seq_seen = msg.obs_seq
        resumed = self._safed
        if self._safed:
            logger.info("WATCHDOG: action resumed (seq=%d) -> clearing safe state", msg.seq)
            self._safed = False
        if self._io is not None:
            if resumed:
                self._io.submit(self._robot_enable_torque)  # safe-stop cut torque; bring it back
            self._io.submit(self._apply_action, msg.goal)  # serial write off the event loop
        else:
            self._apply_action(msg.goal)

    async def _watchdog_loop(self) -> None:
        # Poll at ~4x the timeout so we catch a stall well within one window.
        tick = max(self.action_timeout_s / 4.0, 0.02)
        while not self._stop.is_set():
            await asyncio.sleep(tick)
            stalled = (time.monotonic() - self._last_action_t) > self.action_timeout_s
            if stalled and not self._safed:
                self._safed = True
                self._safe_stop()

    # ----- introspection (tests) -------------------------------------------
    @property
    def is_safed(self) -> bool:
        return self._safed
