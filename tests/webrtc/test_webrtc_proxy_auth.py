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

"""Shared-token auth on the signaling relay."""

import asyncio
import threading

import pytest

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


async def _open_close(url, token):
    sig = WebSocketSignaling(url, "s", "robot", token=token)
    await sig.open()
    await sig.close()


def test_relay_token_gate():
    lt = _Loop()
    runner, port = lt.run(start_relay("127.0.0.1", 0, auth_token="right-token"))
    url = f"ws://127.0.0.1:{port}/ws"
    try:
        # Correct token connects.
        lt.run(_open_close(url, "right-token"))
        # Wrong token is rejected (401 -> handshake error). The signaling client surfaces this
        # as one of several transport errors, so we assert only that the connect fails.
        with pytest.raises(Exception):  # noqa: B017
            lt.run(_open_close(url, "wrong-token"))
        # Missing token is rejected.
        with pytest.raises(Exception):  # noqa: B017
            lt.run(_open_close(url, None))
    finally:
        lt.run(runner.cleanup())
        lt.stop()


def test_no_token_relay_accepts_anyone():
    """Without auth_token configured, the relay is open (back-compat / same-host)."""
    lt = _Loop()
    runner, port = lt.run(start_relay("127.0.0.1", 0))  # no auth_token
    url = f"ws://127.0.0.1:{port}/ws"
    try:
        lt.run(_open_close(url, None))  # connects fine
    finally:
        lt.run(runner.cleanup())
        lt.stop()


def test_full_link_with_token(webrtc_link):
    pytest.importorskip("aiortc", reason="full controller<->daemon link needs aiortc")
    with webrtc_link(token="s3cret") as link:
        assert link.robot.is_connected
        assert link.robot.get_observation()["front"].shape == (48, 64, 3)
