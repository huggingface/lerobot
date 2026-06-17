#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import struct

import numpy as np
import pytest

from lerobot.datasets.episode_video_streaming import assert_hf_hub_range_cache_branch
from lerobot.datasets.mp4 import (
    _box,
    _co64,
    _dinf,
    _hdlr,
    _mdhd,
    _mvhd,
    _stco,
    _stsc_one_sample_per_chunk,
    _stss,
    _stsz,
    _stts,
    _tkhd,
    _vmhd,
    parse_mp4_index,
    synthesize_mp4,
)


def _minimal_mp4(sample_offsets: list[int], *, use_co64: bool = False) -> bytes:
    ftyp = _box(b"ftyp", b"isom\0\0\2\0isomiso2mp41")
    sizes = np.array([10, 10, 10], dtype=np.int64)
    durations = np.array([1000, 1000, 1000], dtype=np.int64)
    stsd_body = struct.pack(">II", 0, 1) + struct.pack(">I4s", 16, b"avc1") + b"\0" * 8
    offsets = _co64(sample_offsets) if use_co64 else _stco(sample_offsets)
    stbl = _box(
        b"stbl",
        _box(b"stsd", stsd_body)
        + _stts(durations)
        + _stsc_one_sample_per_chunk(len(sizes))
        + _stsz(sizes)
        + offsets
        + _stss(np.array([1], dtype=np.int64)),
    )
    minf = _box(b"minf", _vmhd() + _dinf() + stbl)
    mdia = _box(b"mdia", _mdhd(1000, 3000) + _hdlr() + minf)
    trak = _box(b"trak", _tkhd(1, 3000, 64, 48) + mdia)
    moov = _box(b"moov", _mvhd(1000, 3000, 2) + trak)
    mdat_payload_start = 10_000
    free_size = mdat_payload_start - 8 - len(ftyp) - len(moov)
    assert free_size >= 8
    free = _box(b"free", b"\0" * (free_size - 8))
    return ftyp + moov + free + _box(b"mdat", b"x" * 128)


def test_episode_slice_uses_min_max_sample_offsets_for_reordered_chunks():
    mp4 = parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025]))

    sample_slice = mp4.sample_slice(0.0, 2.0, keyframe_pad_s=0, keyframe_pad_fraction=0)

    assert sample_slice.byte_offset == 10_000
    assert sample_slice.byte_length == 60
    assert sample_slice.sample_lo == 0
    assert sample_slice.sample_hi == 2


def test_synthesized_mp4_rebases_one_chunk_per_sample_offsets():
    mp4 = parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025]))
    sample_slice = mp4.sample_slice(0.0, 2.0, keyframe_pad_s=0, keyframe_pad_fraction=0)

    mini = synthesize_mp4(mp4, sample_slice, b"x" * sample_slice.byte_length)
    mini_index = parse_mp4_index("mini.mp4", mini)

    expected = np.array([0, 50, 25], dtype=np.int64) + mini_index.mdat_payload_offset
    np.testing.assert_array_equal(mini_index.sample_offsets, expected)
    np.testing.assert_array_equal(mini_index.sample_sizes, np.array([10, 10, 10]))


def test_parser_accepts_co64_chunk_offsets():
    mp4 = parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025], use_co64=True))

    np.testing.assert_array_equal(mp4.sample_offsets, np.array([10_000, 10_050, 10_025]))


def test_hf_hub_branch_assertion_accepts_requested_revision(monkeypatch):
    class FakeDist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/huggingface/huggingface_hub.git",
                    "vcs_info": {"requested_revision": "feat/hffs-cache-cdn-range-reads"},
                }
            )

    monkeypatch.setattr(
        "lerobot.datasets.episode_video_streaming.metadata.distribution", lambda _: FakeDist()
    )

    assert_hf_hub_range_cache_branch()


def test_hf_hub_branch_assertion_rejects_plain_install(monkeypatch):
    class FakeDist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"url": "https://github.com/huggingface/huggingface_hub.git"})

    monkeypatch.setattr(
        "lerobot.datasets.episode_video_streaming.metadata.distribution", lambda _: FakeDist()
    )

    with pytest.raises(AssertionError):
        assert_hf_hub_range_cache_branch()
