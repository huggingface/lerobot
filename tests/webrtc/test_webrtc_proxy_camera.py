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

"""Camera streaming, preview grab, obs-shape enforcement, and camera plan."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")

from lerobot.robots.webrtc_proxy.capture_agent import CaptureAgent, _fit_frame  # noqa: E402
from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (  # noqa: E402
    WebRTCCameraSpec,
    WebRTCProxyRobotConfig,
)
from lerobot.robots.webrtc_proxy.control import SyntheticInventory  # noqa: E402
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402


class _FakeCamera:
    """Duck-types a lerobot Camera: read_latest() returns a fixed RGB frame."""

    def __init__(self, height: int, width: int, color: tuple[int, int, int]) -> None:
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame[:] = color

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        return self._frame.copy()


# ---- pure units -----------------------------------------------------------
def test_fit_frame_resizes_and_normalizes():
    src = np.zeros((30, 40, 4), dtype=np.uint8)  # wrong size + RGBA
    out = _fit_frame(src, 48, 64)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_apply_camera_plan_updates_size():
    ns = SimpleNamespace(cam_w=64, cam_h=48)
    CaptureAgent._apply_camera_plan(ns, {"width": 32, "height": 24})
    assert (ns.cam_w, ns.cam_h) == (32, 24)


def test_get_observation_enforces_declared_shape():
    """A wrong-sized frame is re-fit to the declared obs shape (no transport needed)."""
    robot = WebRTCProxyRobot(
        WebRTCProxyRobotConfig(cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)})
    )
    robot._connected = True  # bypass the link; exercise get_observation directly
    robot._buffer.add_frame(seq=0, frame=np.full((100, 200, 3), 77, np.uint8))
    robot._buffer.add_state(seq=0, t=1.0, joints={f"{m}.pos": 0.0 for m in robot.motors})
    obs = robot.get_observation()
    assert obs["front"].shape == (48, 64, 3)
    robot._connected = False


# ---- end-to-end over the real link ----------------------------------------
def test_grab_camera_preview_over_link(webrtc_link):
    with webrtc_link(inventory=SyntheticInventory()) as link:
        img = link.robot.grab_camera_preview(1, width=64, height=48)
        assert img.shape == (48, 64, 3)
        assert img.dtype == np.uint8
        other = link.robot.grab_camera_preview(0, width=64, height=48)
        assert not np.array_equal(img, other)  # SyntheticInventory colours each id distinctly


def test_real_camera_frames_reach_the_cloud(webrtc_link):
    color = (200, 100, 50)
    with webrtc_link(camera=_FakeCamera(48, 64, color)) as link:
        frame = link.robot.get_observation()["front"]
        assert frame.shape == (48, 64, 3)
        # VP8 round-trip is lossy; the mean colour must still track the camera frame.
        mean = frame.reshape(-1, 3).mean(axis=0)
        assert np.allclose(mean, color, atol=40), f"got mean {mean}, expected ~{color}"
