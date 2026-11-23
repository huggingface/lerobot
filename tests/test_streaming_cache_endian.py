# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.streaming import sidecar_utils
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import parse_mp4_index
from lerobot.streaming.sidecar import SidecarSpec
from tests.datasets.test_episode_video_streaming import _minimal_mp4


def test_big_endian_timestamps_survive_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    index = parse_mp4_index("endian.mp4", _minimal_mp4([10000, 10050, 10025]))
    index = replace(index, sample_pts=index.sample_pts.astype(">f8"))
    spec = SidecarSpec("test/endian", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    path = tmp_path / "endian.npz"
    EpisodeVideoManifest.save_file_sidecar(
        path, [VideoFileRecord(index.file_path, index.file_size, index)], spec=spec
    )
    cache, payload = sidecar_utils.mapped_sidecar(path)
    timestamp = cache.stat().st_mtime_ns
    reused_cache, reused_payload = sidecar_utils.mapped_sidecar(path)
    assert reused_cache == cache
    assert reused_payload == payload
    assert cache.stat().st_mtime_ns == timestamp
    buffer = np.memmap(cache, mode="r", dtype=np.uint8)
    arrays = sidecar_utils.mapped_arrays(buffer, payload["files"][0])
    np.testing.assert_array_equal(arrays["sample_pts"], index.sample_pts)
    assert arrays["sample_pts"].dtype == np.dtype(">f8")
