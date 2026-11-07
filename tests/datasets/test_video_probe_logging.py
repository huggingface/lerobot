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

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import numpy as np
import pytest

from lerobot.datasets import video_utils


@pytest.mark.parametrize("probe", [video_utils.get_audio_info, video_utils.get_video_info])
@pytest.mark.parametrize("level", [logging.DEBUG, logging.ERROR])
@pytest.mark.parametrize("outcome", ["empty", "error", "metadata"])
def test_metadata_probe_preserves_application_logging(monkeypatch, probe, level, outcome):
    logger = logging.getLogger("libav")
    previous = logger.level
    callback = MagicMock()
    monkeypatch.setattr(video_utils.av.logging, "restore_default_callback", callback)
    stream = SimpleNamespace(
        height=16,
        width=16,
        codec=SimpleNamespace(canonical_name="h264"),
        pix_fmt="yuv420p",
        base_rate=30,
        channels=2,
        bit_rate=128000,
        sample_rate=48000,
        format=SimpleNamespace(bits=16),
        layout=SimpleNamespace(name="stereo"),
    )
    container = MagicMock()
    container.__enter__.return_value.streams = SimpleNamespace(
        audio=[stream] if outcome == "metadata" else [],
        video=[stream] if outcome == "metadata" else [],
    )
    opener = MagicMock(return_value=container)
    if outcome == "error":
        opener.side_effect = OSError("probe failed")
    monkeypatch.setattr(video_utils.av, "open", opener)
    try:
        logger.setLevel(level)
        if outcome == "error":
            with pytest.raises(OSError, match="probe failed"):
                probe("test.mp4")
        else:
            result = probe("test.mp4")
            if outcome == "metadata":
                assert result["has_audio"] is True
        assert logger.level == level
        callback.assert_not_called()
    finally:
        logger.setLevel(previous)


def test_real_video_probe_keeps_metadata_and_logging(tmp_path):
    path = tmp_path / "clip.avi"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("rawvideo", rate=30)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    logger = logging.getLogger("libav")
    previous = logger.level
    try:
        logger.setLevel(logging.ERROR)
        info = video_utils.get_video_info(path)
        assert info["video.fps"] == 30
        assert info["video.width"] == 16
        assert info["video.height"] == 16
        assert info["has_audio"] is False
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(previous)
