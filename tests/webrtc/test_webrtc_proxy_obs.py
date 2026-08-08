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

"""obs / action / watchdog over the real controller<->daemon link (see conftest)."""

import time

import numpy as np
import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")
pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (  # noqa: E402
    WebRTCCameraSpec,
    WebRTCProxyRobotConfig,
)
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402


def test_schema_available_before_connect():
    robot = WebRTCProxyRobot(WebRTCProxyRobotConfig(cameras={"front": WebRTCCameraSpec(48, 64, 30)}))
    assert not robot.is_connected
    assert robot.action_features == {f"{m}.pos": float for m in robot.motors}
    obs_ft = robot.observation_features
    assert obs_ft["front"] == (48, 64, 3)
    assert obs_ft["shoulder_pan.pos"] is float


def test_connect_requires_ws_signaling():
    robot = WebRTCProxyRobot(WebRTCProxyRobotConfig(cameras={"front": WebRTCCameraSpec(48, 64, 30)}))
    with pytest.raises(ValueError):
        robot.connect()  # no signaling_url


def test_observation_roundtrip(webrtc_link):
    with webrtc_link() as link:
        robot = link.robot
        assert robot.is_connected
        obs = robot.get_observation()
        assert set(obs) == set(robot.observation_features)
        frame = obs["front"]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
        for m in robot.motors:
            assert isinstance(obs[f"{m}.pos"], float)
        sent = robot.send_action({"shoulder_pan.pos": 12.5})
        assert sent == {"shoulder_pan.pos": 12.5}


def test_watchdog_safes_on_action_stall_then_clears(webrtc_link):
    with webrtc_link(action_timeout_s=0.3) as link:
        robot, agent = link.robot, link.agent
        # Resume actions -> watchdog clears (it may already have fired during connect).
        robot.send_action({"shoulder_pan.pos": 0.0})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and agent.is_safed:
            time.sleep(0.02)
        assert not agent.is_safed

        # Stop sending -> watchdog must safe within ~action_timeout_s.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not agent.is_safed:
            time.sleep(0.02)
        assert agent.is_safed, "watchdog did not engage after action stall"
