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

"""Closed-loop traceability: action -> obs provenance, and applied feedback back."""

import time

import pytest

from lerobot.robots.webrtc_proxy.protocol import ActionMsg, StateMsg

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")
pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")


# ---- protocol back-compat (new fields are optional) -----------------------
def test_action_msg_obs_seq_roundtrips_and_defaults():
    assert ActionMsg.from_json('{"t":1.0,"seq":2,"goal":{}}').obs_seq == -1  # old senders
    m = ActionMsg(t=1.0, seq=2, goal={"a.pos": 3.0}, obs_seq=7)
    assert ActionMsg.from_json(m.to_json()).obs_seq == 7


def test_state_msg_applied_fields_roundtrip_and_default():
    assert StateMsg.from_json('{"t":1.0,"seq":2,"joints":{}}').applied_seq == -1
    m = StateMsg(t=1.0, seq=2, joints={"a.pos": 1.0}, applied_seq=5, applied_t=9.0)
    back = StateMsg.from_json(m.to_json())
    assert (back.applied_seq, back.applied_t) == (5, 9.0)


# ---- end-to-end over the real link ----------------------------------------
def test_action_provenance_and_applied_feedback(webrtc_link):
    with webrtc_link() as link:
        robot, agent = link.robot, link.agent

        robot.get_observation()  # advances the cloud's _last_obs_seq
        assert robot._last_obs_seq >= 0
        robot.send_action({"shoulder_pan.pos": 1.0})

        # robot received the action stamped with the obs it was derived from.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and agent._last_obs_seq_seen < 0:
            time.sleep(0.02)
        assert agent._last_obs_seq_seen == robot._last_obs_seq

        # Cloud sees the robot report it applied that action (via the state stream).
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and robot._endpoint.last_applied_seq < 1:
            time.sleep(0.02)
        assert robot._endpoint.last_applied_seq >= 1
        assert robot._endpoint.last_applied_t > 0.0
