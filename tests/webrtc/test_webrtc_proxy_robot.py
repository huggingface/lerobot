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

"""M2: a real lerobot Robot drives joints / action / torque through the daemon."""

import time

import numpy as np
import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")
pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import SO100_MOTORS  # noqa: E402


class _FakeBus:
    def __init__(self):
        self.torque = True

    def enable_torque(self):
        self.torque = True

    def disable_torque(self):
        self.torque = False


class _FakeRobot:
    """Duck-types a lerobot so_follower: get_observation / send_action / bus.*torque."""

    def __init__(self, motors, cam_name, height, width, color):
        self.motors = list(motors)
        self.cam_name = cam_name
        self._frame = np.full((height, width, 3), color, np.uint8)
        self.bus = _FakeBus()
        self.last_goal = None

    def get_observation(self):
        obs = {f"{m}.pos": 1.0 for m in self.motors}
        obs[self.cam_name] = self._frame.copy()
        return obs

    def send_action(self, goal):
        self.last_goal = dict(goal)
        return goal


def test_real_robot_drives_obs_action_and_torque(webrtc_link):
    dev = _FakeRobot(SO100_MOTORS, "front", 48, 64, (10, 150, 90))
    with webrtc_link(robot=dev, action_timeout_s=0.3) as link:
        robot = link.robot

        # obs joints + camera come from the robot's own get_observation.
        obs = robot.get_observation()
        assert obs["shoulder_pan.pos"] == 1.0
        assert np.allclose(obs["front"].reshape(-1, 3).mean(0), (10, 150, 90), atol=40)

        # action reaches the robot (applied on the io thread), torque re-enabled.
        robot.send_action({"shoulder_pan.pos": 5.0})
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and dev.last_goal is None:
            time.sleep(0.02)
        assert dev.last_goal == {"shoulder_pan.pos": 5.0}
        assert dev.bus.torque is True

        # stop sending -> watchdog cuts torque (P0 safe stop).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and dev.bus.torque:
            time.sleep(0.02)
        assert dev.bus.torque is False
