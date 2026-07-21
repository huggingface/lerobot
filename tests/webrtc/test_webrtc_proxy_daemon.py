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

"""Two-endpoint daemon link over a real localhost WebSocket signaling relay.

Exercises the production-shaped path (separate event loops, real ws signaling,
WebRTC over localhost host candidates) rather than the in-process loopback. Each
of relay / daemon / cloud-controller runs on its own loop and talks only over
sockets — same as a real robot daemon and a cloud pod.
"""

import asyncio
import threading
import time

import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")
pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (  # noqa: E402
    WebRTCCameraSpec,
    WebRTCProxyRobotConfig,
)
from lerobot.robots.webrtc_proxy.control import SyntheticInventory  # noqa: E402
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402
from lerobot.robots.webrtc_proxy.robot_daemon import run_daemon  # noqa: E402
from lerobot.robots.webrtc_proxy.signaling_server import start_relay  # noqa: E402


class _LoopThread:
    """An asyncio loop running in a daemon thread, for hosting relay/daemon."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


@pytest.fixture
def relay():
    lt = _LoopThread()
    _, port = lt.submit(start_relay("127.0.0.1", 0)).result(timeout=5)
    yield lt, f"ws://127.0.0.1:{port}/ws"
    lt.stop()


@pytest.fixture
def daemon(relay):
    lt, url = relay
    daemon_lt = _LoopThread()
    inv = SyntheticInventory(ports=["/dev/tty.bus-A", "/dev/tty.bus-B"])
    fut = daemon_lt.submit(
        run_daemon(
            url,
            session_id="s1",
            cam_name="front",
            cam_height=48,
            cam_width=64,
            capture_fps=30,
            action_timeout_s=0.5,
            ice_servers=[],
            inventory=inv,
        )
    )
    time.sleep(0.4)  # let the daemon connect + buffer its offer on the relay
    yield url
    fut.cancel()
    time.sleep(0.3)  # let the daemon's finally close its agent + ws session before the loop stops
    daemon_lt.stop()


def _cloud(url: str) -> WebRTCProxyRobot:
    cfg = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        signaling_url=url,
        session_id="s1",
        ice_servers=[],
        capture_fps=30,
        action_timeout_s=0.5,
        connect_timeout_s=20.0,
    )
    return WebRTCProxyRobot(cfg)


def test_cloud_controller_reaches_daemon_over_ws(daemon):
    robot = _cloud(daemon)
    robot.connect()
    try:
        assert robot.is_connected
        obs = robot.get_observation()
        assert set(obs) == set(robot.observation_features)
        assert obs["front"].shape == (48, 64, 3)
        # control plane reaches the daemon's (synthetic) OS inventory
        assert set(robot.list_ports()) == {"/dev/tty.bus-A", "/dev/tty.bus-B"}
        assert robot.send_action({"shoulder_pan.pos": 3.0}) == {"shoulder_pan.pos": 3.0}
    finally:
        robot.disconnect()


def test_daemon_outlives_session_and_serves_again(daemon):
    # First session.
    r1 = _cloud(daemon)
    r1.connect()
    assert r1.get_observation()["front"].shape == (48, 64, 3)
    r1.disconnect()

    # Daemon must have reset and be ready for a brand-new session on the same id.
    r2 = _cloud(daemon)
    r2.connect()
    try:
        assert r2.is_connected
        assert r2.get_observation()["front"].shape == (48, 64, 3)
    finally:
        r2.disconnect()
