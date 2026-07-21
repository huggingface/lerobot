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

"""Pluggable transport layer.

The proxy's logic (capture loop, alignment, watchdog, control-plane RPC, the Robot
API) is transport-agnostic; it only needs:

- named **data channels** (configurable reliability) to send/receive small JSON
  (state, action, control), and
- a one-way **video stream** carrying frames tagged with a capture ``seq``.

``Transport`` is that contract. ``AiortcTransport`` implements it with aiortc
(WebRTC P2P + DataChannels, the default, self-contained backend). A different
backend — e.g. a LiveKit SFU for cross-public-net / scale — can implement the same
interface without touching ``CaptureAgent`` / ``_ProxyEndpoint`` / ``WebRTCProxyRobot``.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from .protocol import VIDEO_CLOCK_RATE, VIDEO_PTS_PER_SEQ, channel_kwargs
from .signaling import Signaling

logger = logging.getLogger(__name__)


def make_transport(
    backend: str,
    *,
    role: str,
    channels: dict[str, bool],
    ice_servers: list[str | dict] | None = None,
    livekit_url: str | None = None,
    livekit_token: str | None = None,
) -> Transport:
    """Build a transport for ``backend`` ("aiortc" | "livekit").

    Both ends of a session MUST use the same backend (an aiortc P2P peer and a LiveKit
    room don't interoperate). "aiortc" is the default, self-contained backend. "livekit"
    (EXPERIMENTAL scaffold, see ``transport_livekit.py``) routes via a LiveKit SFU and
    needs ``livekit_url`` + ``livekit_token``.
    """
    if backend == "aiortc":
        return AiortcTransport(role=role, channels=channels, ice_servers=ice_servers)
    if backend == "livekit":
        if not livekit_url or not livekit_token:
            raise ValueError("transport_backend='livekit' requires livekit_url and livekit_token")
        from .transport_livekit import LiveKitTransport

        return LiveKitTransport(role=role, channels=channels, url=livekit_url, token=livekit_token)
    raise ValueError(f"unknown transport backend {backend!r} (expected 'aiortc' or 'livekit')")


class Channel(ABC):
    """A named message pipe. ``send`` is best-effort (drops if not open)."""

    @abstractmethod
    def send(self, data: str) -> None: ...

    @abstractmethod
    def on_message(self, callback: Callable[[str], None]) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...


class Transport(ABC):
    """Bidirectional transport: named data channels + one video stream.

    Roles: the ``"publisher"`` side offers + sends video; the ``"subscriber"`` side
    answers + receives video. Data channels are bidirectional regardless of role.
    """

    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()

    @abstractmethod
    async def open(self, signaling: Signaling) -> None:
        """Establish the connection (exchange SDP via ``signaling``) and wire channels."""

    @abstractmethod
    def channel(self, label: str) -> Channel:
        """Return the channel handle for ``label`` (created/expected at construction)."""

    @abstractmethod
    def send_frame(self, seq: int, img: np.ndarray) -> None:
        """Publish one video frame tagged with its capture ``seq`` (publisher only)."""

    @abstractmethod
    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """Register ``callback(seq, rgb_ndarray)`` for received frames (subscriber only)."""

    async def wait_closed(self) -> None:
        await self.closed.wait()

    @abstractmethod
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# aiortc backend
# --------------------------------------------------------------------------- #
class _AiortcChannel(Channel):
    """Wraps an aiortc RTCDataChannel, tolerating "registered before it exists"."""

    def __init__(self) -> None:
        self._ch = None
        self._pending_cb: Callable[[str], None] | None = None

    def bind(self, ch) -> None:  # noqa: ANN001 (RTCDataChannel)
        self._ch = ch
        if self._pending_cb is not None:
            ch.on("message", self._pending_cb)

    def send(self, data: str) -> None:
        if self._ch is not None and self._ch.readyState == "open":
            self._ch.send(data)

    def on_message(self, callback: Callable[[str], None]) -> None:
        if self._ch is not None:
            self._ch.on("message", callback)
        else:
            self._pending_cb = callback

    @property
    def is_open(self) -> bool:
        return self._ch is not None and self._ch.readyState == "open"


class _PublisherTrack:
    """An aiortc MediaStreamTrack fed by ``send_frame``; seq rides the frame pts."""

    kind = "video"

    def __init__(self) -> None:
        from aiortc import MediaStreamTrack

        # Build the actual track lazily to avoid importing aiortc at module import.
        class _Track(MediaStreamTrack):
            kind = "video"

            def __init__(self) -> None:
                super().__init__()
                self._q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(maxsize=2)

            def push(self, seq: int, img: np.ndarray) -> None:
                if self._q.full():
                    _ = self._q.get_nowait()  # drop oldest; don't block the capture clock
                self._q.put_nowait((seq, img))

            async def recv(self):
                from av import VideoFrame

                seq, img = await self._q.get()
                frame = VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = seq * VIDEO_PTS_PER_SEQ
                frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
                return frame

        self.track = _Track()

    def push(self, seq: int, img: np.ndarray) -> None:
        self.track.push(seq, img)


class AiortcTransport(Transport):
    """Default backend: WebRTC P2P (media track + DataChannels) over aiortc."""

    def __init__(
        self,
        *,
        role: str,  # "publisher" (offers + sends video) | "subscriber" (answers + recvs video)
        channels: dict[str, bool],  # label -> reliable (reliability honoured by the publisher/offerer)
        # Each entry is a STUN url string or a dict {"urls", ...}. aiortc is direct-UDP:
        # STUN gives a server-reflexive candidate for cross-NAT direct P2P. Static config;
        # the signaling relay also hands out STUN at open(). (The dict form accepts
        # username/credential as a generic escape hatch, but the media-relay path is the
        # LiveKit backend, not a TURN server under aiortc — see DESIGN §11.1.)
        ice_servers: list[str | dict] | None = None,
    ) -> None:
        super().__init__()
        if role not in ("publisher", "subscriber"):
            raise ValueError(f"role must be 'publisher' or 'subscriber', got {role!r}")
        self.role = role
        self._channel_specs = dict(channels)
        self._ice_cfg: list[str | dict] = list(ice_servers or [])
        # The RTCPeerConnection is built in open(), once the signaling relay has had a
        # chance to hand us STUN servers. aiortc fixes iceServers at construction, so we
        # can't build it earlier.
        self.pc = None
        self._channels: dict[str, _AiortcChannel] = {label: _AiortcChannel() for label in channels}
        self._pub = _PublisherTrack() if role == "publisher" else None
        self._frame_cb: Callable[[int, np.ndarray], None] | None = None

    @staticmethod
    def _to_ice_server(cfg: str | dict):  # noqa: ANN205 (RTCIceServer)
        """Coerce a config entry (url string or {urls,username,credential}) to RTCIceServer."""
        from aiortc import RTCIceServer

        if isinstance(cfg, str):
            return RTCIceServer(urls=cfg)
        return RTCIceServer(urls=cfg["urls"], username=cfg.get("username"), credential=cfg.get("credential"))

    def _register(self) -> None:
        @self.pc.on("connectionstatechange")
        async def _on_state() -> None:
            state = self.pc.connectionState
            logger.info("transport connectionState=%s", state)
            if state == "connected":
                self.connected.set()
            elif state in ("failed", "closed", "disconnected"):
                self.closed.set()

        if self.role == "subscriber":

            @self.pc.on("datachannel")
            def _on_channel(ch):  # noqa: ANN001
                if ch.label in self._channels:
                    self._channels[ch.label].bind(ch)

            @self.pc.on("track")
            def _on_track(track):  # noqa: ANN001
                asyncio.ensure_future(self._consume(track))

    async def _consume(self, track) -> None:  # noqa: ANN001
        while True:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("transport: video track ended")
                return
            seconds = float(frame.pts) * float(frame.time_base)
            seq = round(seconds * VIDEO_CLOCK_RATE / VIDEO_PTS_PER_SEQ)
            if self._frame_cb is not None:
                self._frame_cb(seq, frame.to_ndarray(format="rgb24"))

    async def open(self, signaling: Signaling) -> None:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        await signaling.open()
        # Merge static (config) ICE servers with any the relay handed us on connect
        # (e.g. TURN with freshly-minted ephemeral credentials). Built now because aiortc
        # fixes iceServers at RTCPeerConnection construction.
        ice_cfg = self._ice_cfg + list(getattr(signaling, "ice_servers", None) or [])
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[self._to_ice_server(c) for c in ice_cfg])
        )
        self._register()
        if self.role == "publisher":
            for label, reliable in self._channel_specs.items():
                self._channels[label].bind(self.pc.createDataChannel(label, **channel_kwargs(reliable)))
            if self._pub is not None:
                self.pc.addTrack(self._pub.track)
            await self.pc.setLocalDescription(await self.pc.createOffer())
            await signaling.send(self.pc.localDescription)
            answer = await signaling.recv()
            if not isinstance(answer, RTCSessionDescription):
                raise RuntimeError(f"publisher expected an SDP answer, got {type(answer)!r}")
            await self.pc.setRemoteDescription(answer)
        else:
            offer = await signaling.recv()
            if not isinstance(offer, RTCSessionDescription):
                raise RuntimeError(f"subscriber expected an SDP offer, got {type(offer)!r}")
            await self.pc.setRemoteDescription(offer)
            await self.pc.setLocalDescription(await self.pc.createAnswer())
            await signaling.send(self.pc.localDescription)

    def channel(self, label: str) -> Channel:
        return self._channels[label]

    def send_frame(self, seq: int, img: np.ndarray) -> None:
        if self._pub is not None:
            self._pub.push(seq, img)

    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        self._frame_cb = callback

    async def close(self) -> None:
        self.closed.set()
        await self.pc.close()
