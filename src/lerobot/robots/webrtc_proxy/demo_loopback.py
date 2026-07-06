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

"""Single-machine demo of the full WebRTCProxyRobot link.

    uv run python -m lerobot.robots.webrtc_proxy.demo_loopback

Starts the relay + a synthetic robot daemon + the cloud controller as three event
loops in one process (localhost sockets — the same path as a real cloud pod and a
real robot daemon). Then drives it through the synchronous Robot API: control-plane
discovery, re-assembled observation streaming, and the P0 watchdog.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time

from .configuration_webrtc_proxy import WebRTCCameraSpec, WebRTCProxyRobotConfig
from .control import SyntheticInventory
from .proxy_robot import WebRTCProxyRobot
from .robot_daemon import run_daemon
from .signaling_server import start_relay


def _spawn_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()), daemon=True).start()
    return loop


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    relay_loop = _spawn_loop()
    runner, port = asyncio.run_coroutine_threadsafe(start_relay("127.0.0.1", 0), relay_loop).result(timeout=5)
    url = f"ws://127.0.0.1:{port}/ws"

    inv = SyntheticInventory()
    agent_box: dict = {}
    daemon_loop = _spawn_loop()
    daemon_fut = asyncio.run_coroutine_threadsafe(
        run_daemon(
            url,
            "demo",
            cam_name="front",
            cam_height=120,
            cam_width=160,
            capture_fps=30,
            action_timeout_s=0.4,
            ice_servers=[],
            inventory=inv,
            on_agent=lambda a: agent_box.__setitem__("agent", a),
        ),
        daemon_loop,
    )
    time.sleep(0.5)  # let the daemon connect + buffer its offer

    robot = WebRTCProxyRobot(
        WebRTCProxyRobotConfig(
            cameras={"front": WebRTCCameraSpec(height=120, width=160, fps=30)},
            signaling_url=url,
            session_id="demo",
            ice_servers=[],
            capture_fps=30,
            action_timeout_s=0.4,
            connect_timeout_s=20.0,
        )
    )
    print("observation_features:", dict(robot.observation_features))
    print("action_features:    ", robot.action_features)
    robot.connect()

    try:
        print("\n== control plane: cloud-driven device discovery (over the robot daemon) ==")
        print("  list_ports():  ", robot.list_ports())
        before = robot.find_port_begin()
        inv.simulate_unplug(before[-1])  # in production the user unplugs the bus
        print(f"  find_port: begin={before} -> result={robot.find_port_result()!r} (the bus)")

        print("\n== streaming re-assembled observations (capture-ts aligned) ==")
        for _ in range(20):
            obs = robot.get_observation()
            pan = obs["shoulder_pan.pos"]
            print(f"  shoulder_pan.pos={pan:+7.2f}  front={obs['front'].shape}{obs['front'].dtype}")
            robot.send_action({"shoulder_pan.pos": pan + 1.0})
            time.sleep(1 / 15)

        print("\n== stop sending actions; daemon watchdog should SAFE STOP within action_timeout_s ==")
        agent = agent_box.get("agent")
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not (agent and agent.is_safed):
            time.sleep(0.05)
        print(f"  watchdog safed = {agent.is_safed if agent else 'n/a'}")
    finally:
        robot.disconnect()
        daemon_fut.cancel()
        time.sleep(0.3)
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(runner.cleanup(), relay_loop).result(timeout=3)
        daemon_loop.call_soon_threadsafe(daemon_loop.stop)
        relay_loop.call_soon_threadsafe(relay_loop.stop)
        print("\n== disconnected cleanly ==")


if __name__ == "__main__":
    main()
