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

"""ICE config distribution for the aiortc (direct-UDP) backend.

aiortc connects directly: host candidates on a LAN, plus a server-reflexive (public)
candidate via STUN for cross-NAT P2P. The signaling relay hands each peer its STUN
servers on connect, and the transport merges them into the peer connection. (A media
relay across hard NATs is the LiveKit backend's job, not a TURN server under aiortc.)
"""

import asyncio
import threading

import pytest

from lerobot.robots.webrtc_proxy.signaling_server import IceConfig


def test_ice_config_empty_when_no_stun():
    ice = IceConfig()
    assert ice.enabled is False
    assert ice.for_peer("sess:robot") == []


def test_ice_config_distributes_stun():
    ice = IceConfig(stun_urls=["stun:stun.l.google.com:19302", "stun:stun.qq.com:3478"])
    assert ice.enabled is True
    assert ice.for_peer("sess:controller") == [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun.qq.com:3478"]}
    ]


def test_to_ice_server_handles_str_and_dict():
    pytest.importorskip("aiortc", reason="needs the lerobot[webrtc] extra (aiortc)")
    from lerobot.robots.webrtc_proxy.transport import AiortcTransport

    plain = AiortcTransport._to_ice_server("stun:stun.example.com:3478")
    assert plain.username is None
    listed = AiortcTransport._to_ice_server({"urls": ["stun:stun.example.com:3478"]})
    assert listed.urls == ["stun:stun.example.com:3478"]


pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")

from lerobot.robots.webrtc_proxy.signaling import WebSocketSignaling  # noqa: E402
from lerobot.robots.webrtc_proxy.signaling_server import start_relay  # noqa: E402


class _Loop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(
            target=lambda: (asyncio.set_event_loop(self.loop), self.loop.run_forever()), daemon=True
        ).start()

    def run(self, coro, timeout=5):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


def _open_and_read_ice(url: str) -> list:
    lt = _Loop()

    async def _go():
        sig = WebSocketSignaling(url, "sess", role="controller")
        await sig.open()  # consumes the ICE message the relay pushes on join
        servers = sig.ice_servers
        await sig.close()
        return servers

    try:
        return lt.run(_go())
    finally:
        lt.stop()


def test_relay_hands_stun_to_a_client():
    lt = _Loop()
    ice = IceConfig(stun_urls=["stun:stun.l.google.com:19302"])
    try:
        _, port = lt.run(start_relay("127.0.0.1", 0, ice=ice))
        servers = _open_and_read_ice(f"ws://127.0.0.1:{port}/ws")
    finally:
        lt.stop()
    assert servers == [{"urls": ["stun:stun.l.google.com:19302"]}]


def test_relay_hands_empty_list_when_no_stun():
    lt = _Loop()
    try:
        _, port = lt.run(start_relay("127.0.0.1", 0))  # no IceConfig
        servers = _open_and_read_ice(f"ws://127.0.0.1:{port}/ws")
    finally:
        lt.stop()
    assert servers == []
