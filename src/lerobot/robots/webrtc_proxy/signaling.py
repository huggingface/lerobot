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

"""Signaling abstraction: exchanges SDP offer/answer between the two peers.

M1 uses :class:`LoopbackSignaling`, an in-process pair backed by ``asyncio.Queue``
(no STUN/TURN, no network — both peers share one event loop). M3 will add a
WebSocket implementation against the K8s signaler with the *same* ``send``/``recv``
interface, so neither the capture agent nor the proxy changes.

aiortc gathers ICE candidates into the SDP (non-trickle) by the time
``localDescription`` is read, so exchanging full session descriptions is enough on
loopback; trickle-ICE is only needed once real NAT traversal (STUN/TURN) is in play.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from aiortc import RTCSessionDescription


class SignalingClosedError(Exception):
    """Raised by ``recv`` when the peer left or the transport closed."""


class Signaling(Protocol):
    """Minimal duplex SDP exchange used by both endpoints.

    ``ice_servers`` is populated by ``open()`` with any STUN/TURN config the relay hands
    out on connect (empty for loopback / a relay with no TURN configured); the transport
    merges it into the peer connection. See ``signaling_server.py``.
    """

    ice_servers: list

    async def open(self) -> None: ...

    async def send(self, description: RTCSessionDescription) -> None: ...

    async def recv(self) -> RTCSessionDescription: ...

    async def close(self) -> None: ...


class LoopbackSignaling:
    """One end of an in-process signaling pair. See :func:`loopback_signaling_pair`."""

    ice_servers: list = []  # loopback has no relay to hand out STUN/TURN

    def __init__(self, outbox: asyncio.Queue, inbox: asyncio.Queue) -> None:
        self._outbox = outbox
        self._inbox = inbox

    async def open(self) -> None:
        pass

    async def send(self, description: RTCSessionDescription) -> None:
        await self._outbox.put(description)

    async def recv(self) -> RTCSessionDescription:
        return await self._inbox.get()

    async def close(self) -> None:
        pass


def loopback_signaling_pair() -> tuple[LoopbackSignaling, LoopbackSignaling]:
    """Return ``(offerer_signaling, answerer_signaling)`` wired back-to-back."""
    a_to_b: asyncio.Queue = asyncio.Queue()
    b_to_a: asyncio.Queue = asyncio.Queue()
    offerer = LoopbackSignaling(outbox=a_to_b, inbox=b_to_a)
    answerer = LoopbackSignaling(outbox=b_to_a, inbox=a_to_b)
    return offerer, answerer


class WebSocketSignaling:
    """SDP exchange over a WebSocket relay (see ``signaling_server.py``).

    Both the robot daemon (``role="robot"``) and the cloud controller
    (``role="controller"``) connect to the same relay URL with a shared
    ``session`` id; the relay forwards each SDP between them (buffering until the
    peer joins). Non-trickle ICE means just one offer + one answer cross the wire.
    """

    def __init__(self, base_url: str, session: str, role: str, token: str | None = None) -> None:
        sep = "&" if "?" in base_url else "?"
        self._url = f"{base_url}{sep}session={session}&role={role}"
        self._token = token
        self._client = None
        self._ws = None
        self.ice_servers: list = []
        self._pending: dict | None = None  # a non-ICE message read early in open()

    async def open(self) -> None:
        import aiohttp

        self._client = aiohttp.ClientSession()
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        self._ws = await self._client.ws_connect(self._url, headers=headers)
        # The relay pushes an ICE config message on join (before any buffered SDP). Consume
        # it here; if the first message isn't ICE (e.g. an older relay), stash it for recv().
        first = await self._next_message()
        if first is not None and first.get("kind") == "ice":
            self.ice_servers = first.get("iceServers", [])
        else:
            self._pending = first

    async def _next_message(self) -> dict | None:
        """Return the next parsed TEXT message, or None if the socket closed."""
        import aiohttp

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                return msg.json()
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
        return None

    async def send(self, description: RTCSessionDescription) -> None:
        await self._ws.send_json({"kind": "sdp", "type": description.type, "sdp": description.sdp})

    async def recv(self) -> RTCSessionDescription:
        while True:
            data = self._pending if self._pending is not None else await self._next_message()
            self._pending = None
            if data is None:
                raise SignalingClosedError("signaling websocket closed")
            if data.get("kind") == "sdp":
                return RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            if data.get("kind") == "bye":
                raise SignalingClosedError("peer left the signaling session")
            # ignore anything else (e.g. a late ICE refresh) and keep waiting for SDP

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._client is not None:
            await self._client.close()
