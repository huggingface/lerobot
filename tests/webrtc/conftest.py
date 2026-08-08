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

"""Shared fixtures for robot tests.

``webrtc_link``: an in-process relay + robot daemon + cloud controller for testing
WebRTCProxyRobot (which is a pure cloud controller with no in-process loopback).
The relay and a synthetic daemon run as their own event loops in this process and
talk over localhost sockets, exactly as a cloud pod and a robot daemon would.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest


class _LoopThread:
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


class _Link:
    def __init__(self, robot, agent_box, teardown):
        self.robot = robot
        self._agent_box = agent_box
        self._teardown = teardown

    @property
    def agent(self):
        """The daemon's live CaptureAgent (for watchdog / camera-plan assertions)."""
        return self._agent_box.get("agent")

    def close(self):
        self._teardown()


@pytest.fixture
def webrtc_link():
    """Return a context-manager factory yielding a connected ``_Link``.

    with webrtc_link(inventory=..., camera=...) as link:
        link.robot.get_observation(); link.agent.is_safed
    """
    from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (
        WebRTCCameraSpec,
        WebRTCProxyRobotConfig,
    )
    from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot
    from lerobot.robots.webrtc_proxy.robot_daemon import run_daemon
    from lerobot.robots.webrtc_proxy.signaling_server import start_relay

    @contextlib.contextmanager
    def _factory(
        *,
        inventory=None,
        camera=None,
        robot=None,
        reliable_state=False,
        reliable_action=False,
        token=None,
        motors=None,
        cam_name: str = "front",
        height: int = 48,
        width: int = 64,
        fps: int = 30,
        cameras: dict | None = None,  # multi-cam: name -> (height, width); overrides cam_name/size
        action_timeout_s: float = 0.5,
        connect_timeout_s: float = 20.0,
    ):
        relay_lt = _LoopThread()
        runner, port = relay_lt.submit(start_relay("127.0.0.1", 0, auth_token=token)).result(timeout=5)
        url = f"ws://127.0.0.1:{port}/ws"

        # Single camera by default; a `cameras` map enables the multi-camera (tiled) path.
        cam_specs = cameras or {cam_name: (height, width)}

        agent_box: dict = {}
        daemon_lt = _LoopThread()
        daemon_fut = daemon_lt.submit(
            run_daemon(
                url,
                session_id="test",
                motors=motors,
                cam_name=cam_name,
                cam_height=height,
                cam_width=width,
                cameras=cameras,
                capture_fps=fps,
                action_timeout_s=action_timeout_s,
                ice_servers=[],
                inventory=inventory,
                camera=camera,
                robot=robot,
                reliable_state=reliable_state,
                reliable_action=reliable_action,
                signaling_token=token,
                on_agent=lambda a: agent_box.__setitem__("agent", a),
            )
        )
        time.sleep(0.4)  # let the daemon connect + buffer its offer

        cfg_kwargs = {
            "cameras": {n: WebRTCCameraSpec(height=h, width=w, fps=fps) for n, (h, w) in cam_specs.items()},
            "signaling_url": url,
            "session_id": "test",
            "ice_servers": [],
            "capture_fps": fps,
            "action_timeout_s": action_timeout_s,
            "connect_timeout_s": connect_timeout_s,
            "signaling_token": token,
        }
        if motors is not None:
            cfg_kwargs["motors"] = list(motors)
        robot = WebRTCProxyRobot(WebRTCProxyRobotConfig(**cfg_kwargs))

        torn = {"done": False}

        def _teardown():
            if torn["done"]:
                return
            torn["done"] = True
            with contextlib.suppress(Exception):
                robot.disconnect()
            daemon_fut.cancel()
            time.sleep(0.3)
            with contextlib.suppress(Exception):
                relay_lt.submit(runner.cleanup()).result(timeout=3)
            daemon_lt.stop()
            relay_lt.stop()

        robot.connect()
        try:
            yield _Link(robot, agent_box, _teardown)
        finally:
            _teardown()

    yield _factory
