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

"""End-to-end test of the (experimental, optional) LiveKit transport backend.

Skipped by default: it needs the ``livekit`` SDK AND a reachable LiveKit server. To
run it, start a local dev server and point the test at it::

    brew install livekit            # or: see https://docs.livekit.io/home/self-hosting
    livekit-server --dev            # API key/secret default to devkey/secret
    LEROBOT_LIVEKIT_URL=ws://127.0.0.1:7880 \
        LEROBOT_LIVEKIT_API_KEY=devkey LEROBOT_LIVEKIT_API_SECRET=secret \
        pytest tests/webrtc/test_webrtc_proxy_livekit.py -p no:hydra_pytest -sv

LiveKit's FFI client binds one room per process, so the robot daemon (publisher) runs as
a separate subprocess and this test process is the cloud controller (subscriber) —
exactly the real two-process deployment.
"""

import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("livekit", reason="livekit transport needs the lerobot[webrtc-livekit] extra")
# Token signing lives in the separate livekit-api package (livekit.api); guard it too so
# this test SKIPS (not errors) when only `livekit` (rtc) is installed without `livekit-api`.
pytest.importorskip("livekit.api", reason="self-signing needs livekit-api (lerobot[webrtc-livekit])")

_URL = os.environ.get("LEROBOT_LIVEKIT_URL")
if not _URL:
    pytest.skip(
        "set LEROBOT_LIVEKIT_URL (+ LEROBOT_LIVEKIT_API_KEY/SECRET) to run the LiveKit e2e test",
        allow_module_level=True,
    )

_API_KEY = os.environ.get("LEROBOT_LIVEKIT_API_KEY", "devkey")
_API_SECRET = os.environ.get("LEROBOT_LIVEKIT_API_SECRET", "secret")
_ROOM = "lerobot-pytest"

from livekit import api  # noqa: E402

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (  # noqa: E402
    WebRTCCameraSpec,
    WebRTCProxyRobotConfig,
)
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402


def _token(identity: str) -> str:
    return (
        api.AccessToken(_API_KEY, _API_SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=_ROOM))
        .to_jwt()
    )


@pytest.fixture
def daemon():
    """Run the robot daemon (publisher) as its own process, streaming a synthetic source."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lerobot.robots.webrtc_proxy.robot_daemon",
            "--session",
            _ROOM,
            "--transport",
            "livekit",
            "--livekit-url",
            _URL,
            "--livekit-token",
            _token("robot"),
            "--camera-name",
            "front",
            "--width",
            "64",
            "--height",
            "48",
            "--fps",
            "30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)  # let the daemon join the room + start publishing
    assert proc.poll() is None, "daemon exited early"
    yield
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _controller() -> WebRTCProxyRobot:
    cfg = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        transport_backend="livekit",
        livekit_url=_URL,
        livekit_token=_token("controller"),
        connect_timeout_s=25.0,
    )
    return WebRTCProxyRobot(cfg)


def test_controller_reaches_daemon_over_livekit(daemon):
    robot = _controller()
    robot.connect()
    try:
        assert robot.is_connected
        obs = robot.get_observation()
        assert set(obs) == set(robot.observation_features)
        assert obs["front"].shape == (48, 64, 3)

        # The action round-trip lands: the synthetic arm holds the last commanded pose,
        # so the joints we read back converge on what we sent.
        target = 12.0
        for _ in range(20):
            robot.send_action({f"{m}.pos": target for m in robot.motors})
            obs = robot.get_observation()
            time.sleep(0.05)
        assert abs(obs["shoulder_pan.pos"] - target) < 1e-3

        # Observations are advancing (fresh seqs), not a single stuck frame.
        seqs = set()
        for _ in range(10):
            robot.get_observation()
            seqs.add(robot._last_obs_seq)
            time.sleep(0.05)
        assert len(seqs) > 1
    finally:
        robot.disconnect()
