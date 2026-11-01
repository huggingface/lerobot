# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import struct
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import (
    _ctts,
    _has_nonzero,
    _stts,
    parse_mp4_index,
    synthesize_mp4,
    synthesized_mp4_size,
)
from lerobot.streaming.sidecar import SidecarSpec
from tests.datasets.test_episode_video_streaming import _minimal_mp4

UINT32_MAX = 2**32 - 1


class NoDenseList(np.ndarray):
    """Reject expansion of a virtual constant array into Python values."""

    def tolist(self) -> list[int]:
        raise AssertionError("The encoder expands a constant array")


def constant(value: int, count: int, dtype: str = "int64") -> np.ndarray:
    """Create a zero-stride integer array without storing its logical values."""
    return np.broadcast_to(np.array(value, dtype=dtype), (count,))


def test_constant_run_tables_match_dense_reference_at_run_count_boundaries() -> None:
    for count in (0, 1, 2, 255, 65_535, 65_536):
        for value in (0, 7, -7, -(2**31), 2**31 - 1):
            dense = np.full(count, value, dtype=np.int64)
            view = constant(value, count)
            assert _ctts(view) == _ctts(dense)
            assert _has_nonzero(view) == bool(np.any(dense))
        for duration in (0, 1000):
            dense = np.full(count, duration, dtype=np.int64)
            assert _stts(constant(duration, count)) == _stts(dense)


def test_constant_run_table_encoding_uses_bounded_allocation() -> None:
    count = 1_000_000
    durations = constant(1000, count).view(NoDenseList)
    offsets = constant(-7, count).view(NoDenseList)

    tracemalloc.start()
    try:
        duration_table = _stts(durations)
        composition_table = _ctts(offsets)
        peak_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert len(duration_table) <= 32
    assert len(composition_table) <= 32
    assert peak_bytes < durations.nbytes // 16


@pytest.mark.parametrize("table", ["stts", "ctts"])
def test_constant_unsigned_table_encodes_uint32_count_and_value_maximum(table: str) -> None:
    values = constant(UINT32_MAX, UINT32_MAX, dtype="uint64").view(NoDenseList)
    encode = _stts if table == "stts" else _ctts
    expected = struct.pack(">I4sIIII", 24, table.encode(), 0, 1, UINT32_MAX, UINT32_MAX)
    assert encode(values) == expected


def test_constant_signed_table_encodes_uint32_count_and_int32_minimum() -> None:
    values = constant(-(2**31), UINT32_MAX, dtype="int64").view(NoDenseList)
    expected = struct.pack(">I4sIIIi", 24, b"ctts", 1 << 24, 1, UINT32_MAX, -(2**31))
    assert _ctts(values) == expected


@pytest.mark.parametrize("table", ["stts", "ctts"])
def test_constant_table_rejects_count_above_uint32_maximum(table: str) -> None:
    values = constant(1, UINT32_MAX + 1, dtype="uint64").view(NoDenseList)
    encode = _stts if table == "stts" else _ctts
    with pytest.raises(struct.error, match="requires 0 <= number <= 4294967295"):
        encode(values)


@pytest.mark.parametrize("table", ["stts", "ctts"])
def test_constant_table_rejects_unsigned_value_above_uint32_maximum(table: str) -> None:
    values = constant(UINT32_MAX + 1, 2, dtype="uint64").view(NoDenseList)
    encode = _stts if table == "stts" else _ctts
    with pytest.raises(struct.error, match="requires 0 <= number <= 4294967295"):
        encode(values)


@pytest.mark.parametrize("composition_offset", [0, 2, -2])
def test_constant_composition_offsets_preserve_synthesized_bytes_and_size(
    composition_offset: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    index = parse_mp4_index("constant.mp4", _minimal_mp4([10000, 10050, 10025]))
    count = len(index.sample_pts)
    index = replace(
        index,
        sample_durations=constant(1000, count),
        sample_composition_offsets=constant(composition_offset, count),
    )
    dense = replace(
        index,
        sample_durations=np.full(count, 1000, dtype=np.int64),
        sample_composition_offsets=np.full(count, composition_offset, dtype=np.int64),
    )
    spec = SidecarSpec("test/constant", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    path = tmp_path / "constant.npz"
    EpisodeVideoManifest.save_file_sidecar(
        path, [VideoFileRecord(index.file_path, index.file_size, index)], spec=spec
    )
    mapped = EpisodeVideoManifest.load_file_sidecar(path)[index.file_path].mp4
    assert mapped.sample_durations.strides == (0,)
    assert mapped.sample_composition_offsets.strides == (0,)
    assert mapped._has_composition_offsets is (composition_offset != 0)

    span = mapped.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    dense_span = dense.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    payload = b"x" * span.byte_length

    mini = synthesize_mp4(mapped, span, payload)
    dense_mini = synthesize_mp4(dense, dense_span, payload)
    assert synthesized_mp4_size(mapped, span) == len(mini)
    assert synthesized_mp4_size(dense, dense_span) == len(dense_mini)
    assert mini == dense_mini
