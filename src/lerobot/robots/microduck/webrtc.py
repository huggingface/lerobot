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

"""The Microduck head camera over the robot's own WebRTC server, for a client on its network.

``mediad`` encodes the camera as H.264 in hardware and serves it through GStreamer's ``webrtcsink``,
whose signalling server listens on ``ws://<robot>:8443``: the session the robot's ``:8080`` console
opens. Unlike ``GET /frame``, which encodes a full-resolution PNG on the robot's CPU per picture, this
avoids a separate CPU PNG encode and HTTP request for every picture.

The protocol follows Pollen's LAN client (``spaces/policy-playground/lan.py`` in
pollen-robotics/microduck, since replaced by a browser page): the robot offers, this side answers, and
the robot's ``control`` data channel answers ``media.video`` with the camera's mount angle. The robot
serves one session at a time, so its console cannot connect while this one runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _aiortc_available, _websockets_available, require_package

from .camera_frame import CameraFrame

if TYPE_CHECKING or _aiortc_available:
    from aiortc import RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from aiortc.rtcdtlstransport import RTCCertificate
    from aiortc.sdp import candidate_from_sdp

if TYPE_CHECKING or _websockets_available:
    from websockets.asyncio.client import ClientConnection, connect

logger = logging.getLogger(__name__)

CONTROL_LABEL = "control"
# Degrees clockwise the head camera is mounted from upright, assumed if media.video goes unanswered.
DEFAULT_MOUNT_ROTATION = 90


def upright(frame: np.ndarray, rotate: int) -> np.ndarray:
    """Turn a frame from a camera mounted ``rotate`` degrees clockwise from upright."""
    if rotate % 90:
        raise ValueError(f"Camera rotation must be a multiple of 90 degrees, got {rotate}.")
    if rotate % 360 == 0:
        return frame
    # np.rot90 counts counter-clockwise.
    return np.ascontiguousarray(np.rot90(frame, k=-(rotate // 90)))


def patch_aiortc_dtls_ciphers() -> None:
    """Let aiortc agree on a DTLS cipher with ``webrtcsink``; without it DTLS never completes.

    GStreamer endpoints using an RSA certificate need ECDHE-RSA-AES128-GCM-SHA256,
    absent from aiortc's default cipher list (aiortc PR #1392). This is the patch of
    ``reachy_mini.media.central_consumer`` and sets the same flag, so a process importing both patches
    once.
    """
    if getattr(RTCCertificate, "_reachy_cipher_patched", False):
        return
    original = RTCCertificate._create_ssl_context
    ciphers = (
        b"ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-ECDSA-CHACHA20-POLY1305:"
        b"ECDHE-ECDSA-AES128-SHA:ECDHE-ECDSA-AES256-SHA:"
        b"ECDHE-RSA-AES128-GCM-SHA256"
    )

    def patched(self: Any, srtp_profiles: Any) -> Any:
        context = original(self, srtp_profiles)
        context.set_cipher_list(ciphers)
        return context

    RTCCertificate._create_ssl_context = patched
    RTCCertificate._reachy_cipher_patched = True


async def receive_video(
    url: str,
    on_frame: Callable[[CameraFrame], None],
    stop: threading.Event,
    *,
    stall_timeout_s: float,
    rotation_timeout_s: float = 2.0,
) -> None:
    """Pass each upright RGB frame of one session to ``on_frame`` until ``stop`` is set.

    Raises ``ConnectionError`` when the robot refuses or ends the session, or when no frame arrives for
    ``stall_timeout_s``, including while the session starts. The caller decides whether to reconnect.
    """
    require_package("aiortc", extra="microduck")
    require_package("websockets", extra="microduck")
    patch_aiortc_dtls_ciphers()
    session = _Session(on_frame, rotation_timeout_s)

    async def run() -> None:
        async with connect(url, open_timeout=stall_timeout_s, close_timeout=1, max_size=2**20) as websocket:
            await session.run(websocket, stall_timeout_s)

    task = asyncio.create_task(run())
    try:
        while not task.done() and not stop.is_set():
            await asyncio.sleep(0.05)
        if task.done():
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _Session:
    """One camera session: signalling, the peer connection, and the frames it decodes."""

    def __init__(self, on_frame: Callable[[CameraFrame], None], rotation_timeout_s: float):
        self.on_frame = on_frame
        self.rotation_timeout_s = rotation_timeout_s
        self.websocket: ClientConnection | None = None
        self.pc: RTCPeerConnection | None = None
        self.session_id: str | None = None
        self.remote_set = False
        self.sequence = 0
        self.pending_ice: list[dict[str, Any]] = []
        self.rotate: int | None = None
        self.started = time.monotonic()
        # When the session last showed it is alive: its start, then every decoded frame.
        self.alive_at = self.started
        self.failure: Exception | None = None
        self.tasks: set[asyncio.Task] = set()

    async def run(self, websocket: ClientConnection, stall_timeout_s: float) -> None:
        self.websocket = websocket
        reader = asyncio.create_task(self._read())
        try:
            while True:
                if reader.done():
                    reader.result()
                    raise ConnectionError("The robot closed the WebRTC signalling socket.")
                if self.failure is not None:
                    raise self.failure
                if time.monotonic() - self.alive_at > stall_timeout_s:
                    raise ConnectionError(f"No camera frame for {stall_timeout_s:.1f} s.")
                await asyncio.sleep(0.05)
        finally:
            pending = [reader, *self.tasks]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if self.session_id is not None:
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(1):
                        await websocket.send(json.dumps({"type": "endSession", "sessionId": self.session_id}))
            if self.pc is not None:
                await self.pc.close()

    async def _send(self, message: dict[str, Any]) -> None:
        assert self.websocket is not None
        await self.websocket.send(json.dumps(message))

    async def _read(self) -> None:
        assert self.websocket is not None
        async for raw in self.websocket:
            if isinstance(raw, str):
                await self._handle(json.loads(raw))

    async def _handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "welcome":
            await self._send({"type": "list"})
        elif kind == "list":
            producers = message.get("producers") or []
            if not producers:
                raise ConnectionError(
                    "The robot's WebRTC server has no camera producer yet; mediad registers one when "
                    "its pipeline starts."
                )
            # The producer offers, because it knows what it sends; this side only answers.
            await self._send({"type": "startSession", "peerId": producers[0]["id"]})
        elif kind == "sessionStarted":
            self.session_id = message.get("sessionId")
            self._build_peer_connection()
        elif kind == "peer":
            if message.get("sdp"):
                await self._on_offer(message["sdp"])
            elif message.get("ice"):
                if self.remote_set:
                    await self._add_ice(message["ice"])
                else:
                    self.pending_ice.append(message["ice"])
        elif kind == "endSession":
            self.session_id = None
            raise ConnectionError("The robot ended the camera session.")
        elif kind == "error":
            # Typically the robot's one session slot is taken, e.g. by its :8080 console.
            raise ConnectionError(f"The robot's WebRTC server refused the session: {message.get('details')}")

    def _build_peer_connection(self) -> None:
        # No ICE servers: on the robot's own network both sides offer host candidates.
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.pc = pc
        self.remote_set = False

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            if pc.connectionState == "failed":
                self.failure = ConnectionError("The WebRTC connection to the robot failed.")

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "video":
                self._spawn(self._consume(track))

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            if channel.label != CONTROL_LABEL:
                return

            def ask_geometry() -> None:
                channel.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "media.video"}))

            if channel.readyState == "open":
                ask_geometry()
            channel.on("open")(ask_geometry)
            channel.on("message")(self._on_control)

    def _on_control(self, text: str | bytes) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            return
        # The answer to our call, or the notification mediad pushes when the channel opens.
        if message.get("id") == 1:
            video = message.get("result")
        elif message.get("method") == "media.video":
            video = message.get("params")
        else:
            return
        if isinstance(video, dict) and isinstance(video.get("rotate"), int):
            self.rotate = video["rotate"]

    async def _on_offer(self, sdp: dict[str, Any]) -> None:
        if self.pc is None or sdp.get("type") != "offer":
            return
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp["sdp"], type="offer"))
        self.remote_set = True
        pending, self.pending_ice = self.pending_ice, []
        for ice in pending:
            await self._add_ice(ice)
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        # aiortc gathers every candidate before setLocalDescription returns, so none are trickled.
        local = self.pc.localDescription
        await self._send(
            {"type": "peer", "sessionId": self.session_id, "sdp": {"type": local.type, "sdp": local.sdp}}
        )

    async def _add_ice(self, ice: dict[str, Any]) -> None:
        text = (ice.get("candidate") or "").strip()
        if not text or self.pc is None:
            return
        candidate = candidate_from_sdp(text.removeprefix("candidate:"))
        candidate.sdpMid = ice.get("sdpMid")
        if ice.get("sdpMLineIndex") is not None:
            candidate.sdpMLineIndex = int(ice["sdpMLineIndex"])
        try:
            await self.pc.addIceCandidate(candidate)
        except Exception as e:  # nosec B110: one bad candidate does not end the session
            logger.debug(f"Ignored ICE candidate {text!r}: {e}")

    async def _consume(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame = await track.recv()
                self.alive_at = time.monotonic()
                if self.rotate is None:
                    if self.alive_at - self.started < self.rotation_timeout_s:
                        continue
                    logger.warning(
                        f"The robot did not report its camera mount angle; assuming {DEFAULT_MOUNT_ROTATION} "
                        "degrees."
                    )
                    self.rotate = DEFAULT_MOUNT_ROTATION
                self.sequence += 1
                media_time_s = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else None
                )
                self.on_frame(
                    CameraFrame(
                        image=upright(frame.to_ndarray(format="rgb24"), self.rotate),
                        sequence=self.sequence,
                        received_at=self.alive_at,
                        media_time_s=media_time_s,
                        session_id=self.session_id,
                    )
                )
        except MediaStreamError:
            self.failure = ConnectionError("The robot's camera track ended.")
        except Exception as e:
            # A task's exception is otherwise lost; end the session with it instead.
            self.failure = e

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.ensure_future(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
