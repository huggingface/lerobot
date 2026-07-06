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

"""LiveKit transport backend — EXPERIMENTAL (optional).

Implements the :class:`Transport` interface on top of the LiveKit Python SDK
(``livekit`` / ``livekit-rtc``), so the proxy can run over a LiveKit SFU (LiveKit Cloud
or self-hosted) instead of aiortc P2P — which gives built-in signaling + NAT
traversal/TURN + scale, at the cost of running/using a LiveKit server (see DESIGN §11).

Verified end-to-end against a local ``livekit-server --dev`` (obs + action + control
round-trip, fresh aligned observations); aiortc remains the default, CI-tested backend.
Map of concepts:

    our channel label  ->  LiveKit data **topic** (publish_data/on data_received)
    reliable bool      ->  publish_data(reliable=...)
    video stream       ->  a published/subscribed video track
    session_id         ->  room name (in the JWT)
    our Signaling      ->  unused (LiveKit does its own signaling; pass url+token)

Carrying the capture seq with each video frame: LiveKit re-stamps frame timestamps and
``FrameMetadata`` is a fixed proto, so the seq can't ride the frame. The publisher
announces each frame's seq on a reliable data topic (``_SEQ_TOPIC``) and the subscriber
stamps each arriving frame with the latest announced seq (see that constant's note).

Install: ``uv pip install livekit`` (or add the ``webrtc-livekit`` extra).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable

import numpy as np

from .transport import Channel, Transport

logger = logging.getLogger(__name__)


def make_livekit_token(
    *, api_key: str, api_secret: str, identity: str, room: str, ttl_hours: float = 24.0
) -> str:
    """Sign a LiveKit access token (JWT) granting ``identity`` room-join in ``room``.

    Lets each process **self-sign** its own token from a shared API key/secret (so the
    daemon and controller each mint their own ``robot`` / ``controller`` identity instead
    of you pasting two JWTs). Convenient for dev / single-tenant.

    Production note: the API secret can mint a token for *any* room/identity, so don't
    ship it to the user's robot — prefer a cloud token server that hands out scoped,
    short-lived tokens. That's why the transport still accepts a pre-signed token too.
    """
    try:
        from livekit import api
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "signing a LiveKit token needs the LiveKit SDK: `uv pip install livekit-api` "
            "(or the lerobot[webrtc-livekit] extra)."
        ) from e
    from datetime import timedelta

    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_ttl(timedelta(hours=ttl_hours))
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


# LiveKit re-stamps frame timestamps and FrameMetadata is a fixed proto, so the capture
# seq can't ride the frame. The publisher announces each frame's seq on this internal
# reliable data topic. The subscriber tracks the *latest* announced seq and stamps each
# arriving video frame with it.
#
# Why "latest" and not FIFO 1:1: the seq topic (reliable data) delivers every message,
# but the video track is lossy — the SFU/encoder decimates frames (measured ~46% delivery
# on a localhost dev server). A FIFO popleft therefore drifts unboundedly: the Nth
# surviving frame gets the Nth seq, so the newest frame ends up labelled with a seq tens
# of frames in the past (huge stale skew). Stamping with the latest announced seq keeps
# frame seqs monotonic and fresh, and they always pair against a state seq that exists
# (the state stream carries the same seq numbering), with a bounded skew of a frame or two
# — which is what teleop/eval want (freshest coherent obs).
_SEQ_TOPIC = "_seq"


class _LiveKitChannel(Channel):
    """A named data pipe backed by a LiveKit data **topic**."""

    def __init__(self, transport: LiveKitTransport, topic: str, reliable: bool) -> None:
        self._t = transport
        self._topic = topic
        self._reliable = reliable
        self._cb: Callable[[str], None] | None = None

    def send(self, data: str) -> None:
        room = self._t.room
        if room is None or not self._t.connected.is_set():
            return  # best-effort, like the aiortc backend
        # publish_data is async; fire-and-forget on the room's loop.
        asyncio.ensure_future(
            room.local_participant.publish_data(
                data.encode("utf-8"), reliable=self._reliable, topic=self._topic
            )
        )

    def on_message(self, callback: Callable[[str], None]) -> None:
        self._cb = callback

    def _dispatch(self, raw: bytes) -> None:
        if self._cb is not None:
            self._cb(raw.decode("utf-8"))

    @property
    def is_open(self) -> bool:
        return self._t.connected.is_set()


class LiveKitTransport(Transport):
    """Transport over a LiveKit room. EXPERIMENTAL — see module docstring."""

    def __init__(
        self,
        *,
        role: str,  # "publisher" (publishes video) | "subscriber" (subscribes)
        channels: dict[str, bool],  # label -> reliable
        url: str,
        token: str,
    ) -> None:
        super().__init__()
        try:
            from livekit import rtc
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "transport_backend='livekit' needs the LiveKit SDK: `uv pip install livekit` "
                "(or the lerobot[webrtc-livekit] extra)."
            ) from e
        self._rtc = rtc
        if role not in ("publisher", "subscriber"):
            raise ValueError(f"role must be 'publisher' or 'subscriber', got {role!r}")
        self.role = role
        self._url = url
        self._token = token
        self.room = rtc.Room()
        self._channels = {
            label: _LiveKitChannel(self, label, reliable) for label, reliable in channels.items()
        }
        self._frame_cb: Callable[[int, np.ndarray], None] | None = None
        self._video_source = None  # created lazily on first send_frame (needs frame dims)
        self._latest_seq: int | None = None  # subscriber: newest seq announced on _SEQ_TOPIC
        self._consume_task: asyncio.Task | None = None  # subscriber: the VideoStream reader
        self._register()

    def _register(self) -> None:
        rtc = self._rtc

        # NOTE: livekit's Room does NOT emit a "connected" event from connect() — the
        # connection is established synchronously when `await room.connect()` returns
        # (see open()). So we set self.connected there, not via an event handler.
        @self.room.on("disconnected")
        def _on_disconnected(*_args) -> None:
            logger.info("livekit: room disconnected")
            self.closed.set()

        @self.room.on("data_received")
        def _on_data(packet) -> None:  # rtc.DataPacket(data, kind, participant, topic)
            if packet.topic == _SEQ_TOPIC:
                try:
                    seq = int(packet.data.decode("utf-8"))
                except Exception:
                    logger.exception("livekit: bad seq message")
                    return
                # Reliable+ordered, but guard monotonicity defensively.
                if self._latest_seq is None or seq > self._latest_seq:
                    self._latest_seq = seq
                return
            ch = self._channels.get(packet.topic)
            if ch is not None:
                ch._dispatch(packet.data)

        if self.role == "subscriber":

            @self.room.on("track_subscribed")
            def _on_track(track, publication, participant) -> None:  # noqa: ANN001
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._consume_task = asyncio.ensure_future(self._consume(track))

    async def _consume(self, track) -> None:  # noqa: ANN001
        rtc = self._rtc
        stream = rtc.VideoStream(track)
        async for event in stream:  # event.frame: rtc.VideoFrame
            if self._latest_seq is None:
                continue  # no seq announced yet (startup); drop until we can label frames
            seq = self._latest_seq  # freshest announced seq (see _SEQ_TOPIC note)
            frame = event.frame.convert(rtc.VideoBufferType.RGB24)
            img = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            if self._frame_cb is not None:
                self._frame_cb(seq, np.ascontiguousarray(img))

    async def open(self, signaling=None) -> None:  # noqa: ANN001 - signaling unused for LiveKit
        await self.room.connect(self._url, self._token)
        # connect() establishes the link synchronously and does NOT fire a "connected"
        # event, so mark connected here. auto_subscribe is on by default, so the
        # subscriber's track_subscribed fires for the publisher's video track.
        self.connected.set()
        # publish_data/track work after connect; the video track is published lazily on
        # the first send_frame so we know the frame dimensions.

    def channel(self, label: str) -> Channel:
        return self._channels[label]

    def send_frame(self, seq: int, img: np.ndarray) -> None:
        if self.role != "publisher":
            return
        rtc = self._rtc
        h, w = img.shape[:2]
        if self._video_source is None:
            self._video_source = rtc.VideoSource(w, h)
            track = rtc.LocalVideoTrack.create_video_track("camera", self._video_source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            asyncio.ensure_future(self.room.local_participant.publish_track(track, options))
        # Announce this frame's seq (reliable, ordered) so the subscriber can stamp the
        # frame with the freshest seq (see the _SEQ_TOPIC note for why not FIFO 1:1).
        asyncio.ensure_future(
            self.room.local_participant.publish_data(str(seq).encode(), reliable=True, topic=_SEQ_TOPIC)
        )
        frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, np.ascontiguousarray(img).tobytes())
        self._video_source.capture_frame(frame)

    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        self._frame_cb = callback

    async def close(self) -> None:
        self.closed.set()
        if self._consume_task is not None:
            self._consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._consume_task
        await self.room.disconnect()
