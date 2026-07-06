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

"""Simulate the *remote* control plane on one machine.

Spins up the relay + robot daemon + cloud controller as three independent event
loops in this single process — they talk only over localhost sockets, exactly as a
real cloud pod and a real robot daemon would over the public internet (only the IP
differs). Then it runs one control-plane RPC from the controller and prints what
came back "from the robot".

    python -m lerobot.robots.webrtc_proxy.sim_remote --rpc list_cameras
    python -m lerobot.robots.webrtc_proxy.sim_remote --rpc list_ports
    python -m lerobot.robots.webrtc_proxy.sim_remote --rpc find_port
    python -m lerobot.robots.webrtc_proxy.sim_remote --rpc observe
    python -m lerobot.robots.webrtc_proxy.sim_remote --rpc all

Default uses synthetic devices (no hardware needed). ``--real-devices`` makes the
daemon enumerate this machine's actual ports/cameras via ``LocalDeviceInventory``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import threading
import time

from .configuration_webrtc_proxy import WebRTCCameraSpec, WebRTCProxyRobotConfig
from .control import DeviceInventory, LocalDeviceInventory, SyntheticInventory
from .proxy_robot import WebRTCProxyRobot
from .robot_daemon import run_daemon
from .signaling_server import start_relay

logger = logging.getLogger(__name__)

# Example synthetic devices (ids mirror real output: opencv index / realsense serial).
_SIM_PORTS = ["/dev/tty.usbmodem-bus-A", "/dev/tty.usbmodem-bus-B"]
_SIM_CAMERAS = [
    {"type": "opencv", "id": 0, "name": "FaceTime HD Camera", "width": 1280, "height": 720, "fps": 30.0},
    {"type": "opencv", "id": 1, "name": "Logitech C920", "width": 640, "height": 480, "fps": 30.0},
]


def _spawn_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()), daemon=True).start()
    return loop


def _run_rpc(
    robot: WebRTCProxyRobot, rpc: str, inventory: DeviceInventory, camera_id, save: str | None
) -> None:
    if rpc == "grab":
        img = robot.grab_camera_preview(camera_id, width=320, height=240)
        print(
            f"REMOTE grab_camera_preview(id={camera_id!r}): {img.shape} {img.dtype} "
            f"mean RGB {img.reshape(-1, 3).mean(0).round(1)}"
        )
        if save:
            from PIL import Image

            Image.fromarray(img).save(save)
            print(f"  saved {save}")
        return
    if rpc in ("list_ports", "all"):
        print("REMOTE list_ports():", robot.list_ports())
    if rpc in ("list_cameras", "all"):
        print("REMOTE list_cameras():")
        for cam in robot.list_cameras():
            print("  ", cam)
    if rpc in ("observe", "all"):
        obs = robot.get_observation()
        frame = obs[robot.cam_name]
        print(f"REMOTE get_observation(): {sorted(k for k in obs if k.endswith('.pos'))}")
        print(f"  {robot.cam_name}: {frame.shape} {frame.dtype}")
    if rpc in ("find_port", "all"):
        before = robot.find_port_begin()
        print("REMOTE find_port_begin():", before)
        if isinstance(inventory, SyntheticInventory):
            inventory.simulate_unplug(before[-1])  # headless: auto "unplug" the last port
            print(f"  (simulated unplug of {before[-1]})")
        else:
            input("  Unplug the motor-bus USB on this machine, then press Enter... ")
        print("REMOTE find_port_result():", robot.find_port_result())


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a remote control-plane RPC on one machine")
    parser.add_argument(
        "--rpc",
        default="list_cameras",
        choices=["list_ports", "list_cameras", "grab", "find_port", "observe", "all"],
    )
    parser.add_argument("--real-devices", action="store_true", help="enumerate this machine's real devices")
    parser.add_argument("--camera-id", default="0", help="camera id for --rpc grab (int index or path)")
    parser.add_argument("--save", default=None, help="for --rpc grab: save the frame to this PNG path")
    args = parser.parse_args()
    camera_id = int(args.camera_id) if args.camera_id.isdigit() else args.camera_id
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    inventory: DeviceInventory = (
        LocalDeviceInventory()
        if args.real_devices
        else SyntheticInventory(ports=list(_SIM_PORTS), cameras=[dict(c) for c in _SIM_CAMERAS])
    )

    # cloud: signaling relay
    relay_loop = _spawn_loop()
    runner, port = asyncio.run_coroutine_threadsafe(start_relay("127.0.0.1", 0), relay_loop).result(timeout=5)
    url = f"ws://127.0.0.1:{port}/ws"

    # robot: daemon
    daemon_loop = _spawn_loop()
    daemon_fut = asyncio.run_coroutine_threadsafe(
        run_daemon(
            url,
            "sim",
            cam_name="front",
            cam_height=48,
            cam_width=64,
            capture_fps=30,
            ice_servers=[],
            inventory=inventory,
        ),
        daemon_loop,
    )
    time.sleep(0.5)  # let the daemon connect + buffer its offer

    # cloud: controller
    robot = WebRTCProxyRobot(
        WebRTCProxyRobotConfig(
            cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
            signaling_url=url,
            session_id="sim",
            ice_servers=[],
            connect_timeout_s=20.0,
        )
    )
    print(f"controller connecting to {url} ...")
    robot.connect()
    try:
        _run_rpc(robot, args.rpc, inventory, camera_id, args.save)
    finally:
        # Ordered, graceful shutdown so aiohttp/aiortc close cleanly (no pending-task spam):
        robot.disconnect()  # closes the controller pc + its ws session
        daemon_fut.cancel()  # interrupts the daemon's wait; its finally closes agent + ws
        time.sleep(0.3)  # let that finally run on the still-spinning daemon loop
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(runner.cleanup(), relay_loop).result(timeout=3)
        daemon_loop.call_soon_threadsafe(daemon_loop.stop)
        relay_loop.call_soon_threadsafe(relay_loop.stop)
        time.sleep(0.1)
    print("done")


if __name__ == "__main__":
    main()
