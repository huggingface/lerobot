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

"""Configurable channel reliability (control always reliable; state/action per use case)."""

import pytest

from lerobot.robots.webrtc_proxy.protocol import RELIABLE_KWARGS, UNRELIABLE_KWARGS, channel_kwargs

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")
pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")


def test_channel_kwargs_helper():
    assert channel_kwargs(True) == RELIABLE_KWARGS == {"ordered": True}
    assert channel_kwargs(False) == UNRELIABLE_KWARGS == {"ordered": False, "maxRetransmits": 0}


def _ch(agent, label):
    # the underlying aiortc RTCDataChannel behind the transport's Channel handle
    return agent._transport._channels[label]._ch


def test_default_realtime_unreliable_state_action(webrtc_link):
    # teleop/eval default: fresh, drop-on-loss.
    with webrtc_link() as link:
        a = link.agent
        assert _ch(a, "state").ordered is False and _ch(a, "state").maxRetransmits == 0
        assert _ch(a, "action").ordered is False and _ch(a, "action").maxRetransmits == 0
        assert _ch(a, "control").ordered is True  # control always reliable+ordered


def test_record_profile_reliable_state_and_action(webrtc_link):
    # record: both state and action reliable so no obs/action is lost from a transition.
    with webrtc_link(reliable_state=True, reliable_action=True) as link:
        a = link.agent
        assert _ch(a, "state").ordered is True
        assert _ch(a, "action").ordered is True
        assert _ch(a, "control").ordered is True
