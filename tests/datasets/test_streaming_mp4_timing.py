# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Source-decoder pixel oracles for episode MP4 presentation timing."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from lerobot.streaming.episode_cache import EpisodeByteCache
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import (
    _box,
    _full_box,
    _parse_edit_offset,
    parse_mp4_index,
    synthesize_mp4,
    synthesized_mp4_size,
)
from lerobot.streaming.sidecar import SidecarSpec

av = pytest.importorskip("av")


@pytest.fixture(
    params=[
        (0, False, 0, False),
        (3, False, 0, False),
        (3, True, 0, False),
        (3, False, 0.5, False),
        (3, False, 0, True),
    ]
)
def encoded_video(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, dict[int, NDArray[np.uint8]]]:
    b_frames, signed_offsets, start_offset, open_gop = request.param
    path = tmp_path / "source.mp4"
    options = {"movflags": "+negative_cts_offsets"} if signed_offsets else {}
    if start_offset:
        options["output_ts_offset"] = str(start_offset)
    with av.open(str(path), "w", options=options) as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"crf": "18", "g": "30", "bf": str(b_frames), "b_strategy": "0"}
        if open_gop:
            stream.options["x264-params"] = "open-gop=1:scenecut=0:keyint=30:min-keyint=30"
        for frame_index in range(60):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), frame_index * 4, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        expected = {
            round(float(frame.pts * frame.time_base) * 30): frame.to_ndarray(format="rgb24")
            for frame in source.decode(video=0)
        }
    assert len(expected) == 60
    return path, expected


@pytest.mark.parametrize(
    "start,end", [(0.0, 0.2), (0.9, 1.2), (1.1, 1.4), (1.7, 1.99), (29 / 30, 29 / 30), (1.0, 1.0)]
)
@pytest.mark.parametrize("backend", ["pyav", "torchcodec"])
def test_episode_synthesis_preserves_source_timestamp_pixels(
    encoded_video: tuple[Path, dict[int, NDArray[np.uint8]]],
    start: float,
    end: float,
    backend: str,
    tmp_path: Path,
) -> None:
    if backend == "torchcodec":
        pytest.importorskip("torchcodec")
    path, expected = encoded_video
    source_origin = min(expected) / 30
    start, end = start + source_origin, end + source_origin
    data = path.read_bytes()
    index = parse_mp4_index(path.name, data)
    # Exercise persisted indices as well: loading a sidecar must retain all timing information.
    sidecar_path = tmp_path / "sidecar.npz"
    spec = SidecarSpec("test/dataset", "fixed-revision", str(tmp_path), ((path.name, len(data)),))
    EpisodeVideoManifest.save_file_sidecar(
        sidecar_path, [VideoFileRecord(path.name, len(data), index)], spec=spec
    )
    index = EpisodeVideoManifest.load_file_sidecar(sidecar_path)[path.name].mp4
    sample_slice = index.sample_slice(start, end, keyframe_pad_s=0, keyframe_pad_fraction=0)
    mini = synthesize_mp4(
        index,
        sample_slice,
        data[sample_slice.byte_offset : sample_slice.byte_offset + sample_slice.byte_length],
    )
    assert synthesized_mp4_size(index, sample_slice) == len(mini)
    with av.open(io.BytesIO(mini)) as rebuilt:
        actual = {
            round(
                (float(frame.pts * frame.time_base) + sample_slice.source_start_pts) * 30
            ): frame.to_ndarray(format="rgb24")
            for frame in rebuilt.decode(video=0)
        }
    requested = list(range(round(start * 30), min(int(end * 30) + 1, max(expected) + 1)))
    assert requested
    for frame_index in requested:
        np.testing.assert_array_equal(actual[frame_index], expected[frame_index])

    values = {
        "file_id": 0,
        "mdat_offset": sample_slice.byte_offset,
        "mdat_length": sample_slice.byte_length,
        "first_pts": start,
        "last_pts": end,
        "frame_count": sample_slice.sample_hi - sample_slice.sample_lo + 1,
        "sample_lo": sample_slice.sample_lo,
        "sample_hi": sample_slice.sample_hi,
        "source_start_pts": sample_slice.source_start_pts,
    }
    manifest = EpisodeVideoManifest(
        video_keys=["camera"],
        files=[VideoFileRecord(path.name, len(data), index)],
        spans={key: np.full((1, 1), value) for key, value in values.items()},
    )
    # Reverse order and repeat a frame to exercise the actual temporal-query contract.
    requested = list(reversed(requested)) + [requested[0]]
    if backend == "torchcodec":
        from torchcodec.decoders import VideoDecoder

        # Backends can differ by one RGB quantization level; use the original
        # source with the same backend, independently of the mini-MP4 index.
        decoder = VideoDecoder(str(path))
        source_pixels = decoder.get_frames_at(indices=[frame - min(expected) for frame in requested]).data
    else:
        source_pixels = None
    with EpisodeByteCache(manifest, tmp_path, video_backend=backend) as cache:
        pixels = cache.get_frames(0, "camera", [frame_index / 30 for frame_index in requested])
        assert cache.decoder_fallback_count == 0
    for position, (frame_index, actual_frame) in enumerate(zip(requested, pixels, strict=True)):
        expected_frame = (
            source_pixels[position].permute(1, 2, 0).numpy()
            if source_pixels is not None
            else expected[frame_index]
        )
        np.testing.assert_array_equal(actual_frame.permute(1, 2, 0).numpy(), expected_frame)


@pytest.mark.parametrize("version", [0, 1])
def test_edit_list_timescales_and_encoder_delay(version: int) -> None:
    layout = ">Iihh" if version == 0 else ">Qqhh"
    # 0.5 seconds empty time followed by a 2-frame encoder delay at 30 fps.
    entries = struct.pack(layout, 500, -1, 1, 0) + struct.pack(layout, 2000, 1024, 1, 0)
    trak = _box(b"edts", _full_box(b"elst", version, 0, struct.pack(">I", 2) + entries))
    assert _parse_edit_offset(trak, 1000, 15360) == pytest.approx(0.5 - 2 / 30)


@pytest.mark.parametrize("entries", [[(1000, 0, 2, 0)], [(1000, 0, 1, 0), (1000, 0, 1, 0)]])
def test_unsupported_edit_timeline_fails_explicitly(entries: list[tuple[int, int, int, int]]) -> None:
    payload = struct.pack(">I", len(entries)) + b"".join(struct.pack(">Iihh", *entry) for entry in entries)
    trak = _box(b"edts", _full_box(b"elst", 0, 0, payload))
    with pytest.raises(ValueError, match="Unsupported MP4 edit list"):
        _parse_edit_offset(trak, 1000, 15360)
