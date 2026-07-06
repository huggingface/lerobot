#!/usr/bin/env python

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

"""robot side: serve a real SO-100 over WebRTC.

Run this on the machine the arm + camera are plugged into. It opens the real
``SO100Follower`` (serial bus + camera, torque enabled), then hands it to the
WebRTCProxyRobot daemon: the daemon streams joints + camera to the cloud, applies
incoming actions to the motors, and — crucially — cuts torque if the action stream
stalls (network drop / cloud crash), so the arm never holds a dangerous pose.

    Onboarding first (find the ids — see README):
        uv run lerobot-find-port            # -> PORT below
        uv run lerobot-find-cameras         # -> CAMERA_INDEX below

    Then:
        uv run python examples/webrtc_remote_so100/robot_daemon_so100.py
"""

import asyncio

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.webrtc_proxy.robot_daemon import run_daemon

# ── fill these in from onboarding ──────────────────────────────────────────
PORT = "/dev/tty.usbmodem5A460814411"  # lerobot-find-port
CAMERA_INDEX = 0  # lerobot-find-cameras
SIGNALING_URL = "ws://127.0.0.1:8765/ws"  # the relay (cloud); 127.0.0.1 for a same-host test
SESSION_ID = "so100"
FPS, WIDTH, HEIGHT = 30, 640, 480
# Public internet? add STUN/TURN urls here (M4); empty is fine same-host / same-LAN.
ICE_SERVERS: list[str] = []


def main() -> None:
    robot = SO100Follower(
        SO100FollowerConfig(
            port=PORT,
            id="webrtc_so100",
            cameras={
                "front": OpenCVCameraConfig(index_or_path=CAMERA_INDEX, width=WIDTH, height=HEIGHT, fps=FPS)
            },
        )
    )
    robot.connect()  # opens the bus + camera and enables torque
    try:
        # run_daemon outlives any single cloud session: serve one, safe the arm on
        # drop, then wait for the next. The robot is reused across sessions.
        asyncio.run(
            run_daemon(
                SIGNALING_URL,
                session_id=SESSION_ID,
                motors=list(robot.bus.motors),
                cam_name="front",
                cam_height=HEIGHT,
                cam_width=WIDTH,
                capture_fps=FPS,
                action_timeout_s=0.5,
                ice_servers=ICE_SERVERS,
                robot=robot,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()  # disables torque on disconnect


if __name__ == "__main__":
    main()
