# Copyright 2024-2026 The LeRobot Authors.
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

"""Regression tests for video timestamp precision (issue #3580).

Video frame timestamps were stored with the ``VideoFrame`` feature as float32.
Once accumulated footage offsets grow (long recordings, or concatenating many
episodes), float32 no longer represents sub-frame increments — its ulp exceeds
the frame interval — so adjacent frames collapse onto the same timestamp and
frame boundaries drift. Timestamps must be float64.

The large offsets below stand in for a concatenated/long dataset; they trigger
the same rounding collapse without materializing millions of rows.
"""

import numpy as np
import pyarrow as pa
import pytest
from datasets import Dataset, Features

from lerobot.datasets.video_utils import VideoFrame

FPS = 30
# ~7 days of accumulated footage: at this magnitude the float32 ulp is 1/16 s,
# which already exceeds the 1/30 s frame interval.
OFFSET_S = 600_000.0
NUM_FRAMES = 10 * FPS


def _make_timestamps() -> np.ndarray:
    return OFFSET_S + np.arange(NUM_FRAMES, dtype=np.float64) / FPS


def test_video_frame_timestamp_is_float64():
    """The stored Arrow schema must use float64 for timestamps."""
    ts_field = VideoFrame().pa_type.field("timestamp")
    assert ts_field.type == pa.float64(), f"expected pa.float64(), got {ts_field.type}"


def test_video_frame_storage_preserves_monotonicity():
    """Storing large-offset frame timestamps via the VideoFrame feature must
    keep them strictly increasing."""
    timestamps = _make_timestamps()

    features = Features({"observation.video": VideoFrame()})
    data = {
        "observation.video": [
            {"path": "videos/episode_0.mp4", "timestamp": t} for t in timestamps
        ]
    }
    ds = Dataset.from_dict(data, features=features)

    stored_ts = np.array(
        [frame["timestamp"] for frame in ds.with_format("numpy")["observation.video"]],
        dtype=np.float64,
    )
    assert np.all(np.diff(stored_ts) > 0), "timestamps collapsed during storage"


def test_float32_reference_shows_the_issue():
    """Sanity check: float32 storage actually breaks monotonicity at this scale.

    Documents the failure mode the float64 fix addresses; remove if float32
    ever becomes precise enough here.
    """
    ts32 = _make_timestamps().astype(np.float32)
    assert not np.all(np.diff(ts32.astype(np.float64)) > 0)


if __name__ == "__main__":
    pytest.main([__file__])
