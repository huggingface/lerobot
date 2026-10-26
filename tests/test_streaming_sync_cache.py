# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.streaming import sidecar_utils
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import parse_mp4_index, synthesize_mp4
from lerobot.streaming.sidecar import SidecarSpec

av = pytest.importorskip("av")


@pytest.fixture
def real_video(tmp_path: Path) -> tuple[Path, bytes]:
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=3)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        stream.options = {"bf": "0", "g": "3"}
        for position in range(3):
            pixels = np.full((32, 32, 3), 30 + 70 * position, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        frames = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]
    assert len(frames) == 3
    return path, b"".join(frame.tobytes() for frame in frames)


def test_forged_scalar_sync_count_recreates_real_video_cache_before_lookup(
    real_video: tuple[Path, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, expected_pixels = real_video
    reference = parse_mp4_index(source.name, source.read_bytes())
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    spec = SidecarSpec("test/sync-corruption", "rev", str(tmp_path), ((source.name, source.stat().st_size),))
    path = tmp_path / "index.npz"
    EpisodeVideoManifest.write_file_sidecar(
        path, [source.name], tmp_path, spec=spec, range_backend="fsspec", workers=1
    )
    cache, payload = sidecar_utils.mapped_sidecar(path)
    original = cache.read_bytes()
    length = int.from_bytes(original[-16:-8], "little")
    descriptor = payload["files"][0]["arrays"]["sync_samples"]
    descriptor[1:] = [2**60, "<u1", 1]
    metadata = json.dumps(payload).encode()
    forged = original[: -16 - length] + metadata + len(metadata).to_bytes(8, "little") + original[-8:]
    cache.write_bytes(forged)

    # Fail before lookup on the old implementation, without a huge allocation attempt.
    with pytest.raises(ValueError, match="Invalid mapped MP4 index array"):
        sidecar_utils._read_metadata(cache)
    index = EpisodeVideoManifest.load_file_sidecar(path)[source.name].mp4
    assert cache.read_bytes() == original
    assert len(index.sample_pts) == 3
    np.testing.assert_array_equal(index.sync_samples, reference.sync_samples)
    span = index.sample_slice(0, 1, keyframe_pad_s=0, keyframe_pad_fraction=0)
    raw = source.read_bytes()
    mini = synthesize_mp4(index, span, raw[span.byte_offset : span.byte_offset + span.byte_length])
    with av.open(io.BytesIO(mini)) as rebuilt:
        actual = b"".join(frame.to_ndarray(format="rgb24").tobytes() for frame in rebuilt.decode(video=0))
    assert actual == expected_pixels


def test_duplicate_sync_entries_keep_dense_storage_and_exact_archive_values(
    real_video: tuple[Path, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _pixels = real_video
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    raw = source.read_bytes()
    index = replace(parse_mp4_index(source.name, raw), sync_samples=np.array([0, 0, 0], dtype=np.int64))
    spec = SidecarSpec("test/sync-duplicates", "rev", str(tmp_path), ((source.name, len(raw)),))
    path = tmp_path / "duplicates.npz"
    EpisodeVideoManifest.save_file_sidecar(path, [VideoFileRecord(source.name, len(raw), index)], spec=spec)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["0/sync_samples"].dtype == np.int64
        np.testing.assert_array_equal(archive["0/sync_samples"], [0, 0, 0])
    cache, payload = sidecar_utils.mapped_sidecar(path)
    assert payload["files"][0]["arrays"]["sync_samples"][1] == 3
    assert payload["files"][0]["arrays"]["sync_samples"][3] == 3
    sidecar_utils._read_metadata(cache)
    loaded = EpisodeVideoManifest.load_file_sidecar(path)[source.name].mp4
    np.testing.assert_array_equal(loaded.sync_samples, [0, 0, 0])
    assert loaded.sync_samples.strides == (loaded.sync_samples.dtype.itemsize,)
    assert not loaded.sync_samples.flags.writeable
    loaded.sample_slice(0, 1, keyframe_pad_s=0, keyframe_pad_fraction=0)
