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

"""Multi-camera over a single video track via tiling (stack on the robot, slice on cloud)."""

import numpy as np
import pytest

from lerobot.robots.webrtc_proxy import tiling


def _frame(h, w, val):
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_tile_untile_roundtrip_different_sizes():
    specs = tiling.ordered_specs({"wrist": (24, 32), "front": (48, 64), "top": (16, 64)})
    # sorted by name: front, top, wrist
    assert [s[0] for s in specs] == ["front", "top", "wrist"]
    frames = {"front": _frame(48, 64, 10), "top": _frame(16, 64, 20), "wrist": _frame(24, 32, 30)}

    combined = tiling.tile(frames, specs)
    assert combined.shape == (48 + 16 + 24, 64, 3)  # stacked heights x max width

    out = tiling.untile(combined, specs)
    assert set(out) == {"front", "top", "wrist"}
    for name in frames:
        assert np.array_equal(out[name], frames[name])  # exact, lossless slice-back


def test_single_camera_tiles_to_identity():
    specs = tiling.ordered_specs({"front": (48, 64)})
    f = _frame(48, 64, 7)
    combined = tiling.tile({"front": f}, specs)
    assert combined.shape == (48, 64, 3)
    assert np.array_equal(tiling.untile(combined, specs)["front"], f)


def test_narrower_camera_is_width_padded_then_recovered():
    specs = tiling.ordered_specs({"a": (10, 64), "b": (10, 32)})  # b is narrower
    frames = {"a": _frame(10, 64, 1), "b": _frame(10, 32, 2)}
    combined = tiling.tile(frames, specs)
    assert combined.shape == (20, 64, 3)
    out = tiling.untile(combined, specs)
    assert np.array_equal(out["b"], frames["b"])  # the narrower one slices back to 32 wide


# --- end-to-end over the real loopback link (relay + daemon + cloud) -----------
pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")
pytest.importorskip("aiohttp", reason="signaling needs aiohttp (lerobot[webrtc])")


def test_three_cameras_stream_and_split(webrtc_link):
    cameras = {"front": (48, 64), "wrist": (32, 48), "top": (24, 64)}
    with webrtc_link(cameras=cameras) as link:
        feats = link.robot.observation_features
        for name, (h, w) in cameras.items():
            assert feats[name] == (h, w, 3)

        obs = link.robot.get_observation()
        # all three cameras present, each at its declared shape
        for name, (h, w) in cameras.items():
            assert obs[name].shape == (h, w, 3)
            assert obs[name].dtype == np.uint8

        # synthetic frames are coloured distinctly per camera (B channel = idx*80 in
        # name-sorted order: front=0, top=1, wrist=2) — proves the slices aren't crossed.
        # VP8 is lossy, so check near the expected value (not exact).
        blues = {name: float(np.median(obs[name][..., 2])) for name in sorted(cameras)}
        assert abs(blues["front"] - 0) <= 8
        assert abs(blues["top"] - 80) <= 8
        assert abs(blues["wrist"] - 160) <= 8
