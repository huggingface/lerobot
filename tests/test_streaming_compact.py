# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.streaming import sidecar_utils
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import (
    compact_integer_array,
    parse_mp4_index,
    synthesize_mp4,
    synthesized_mp4_size,
)
from lerobot.streaming.sidecar import SidecarSpec
from tests.datasets.test_episode_video_streaming import _minimal_mp4


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([0, 255], "uint8"),
        ([0, 256, 65535], "uint16"),
        ([0, 65536, 2**32 - 1], "uint32"),
        ([0, 2**32, 2**63 - 1], "int64"),
        ([-128, 127], "int8"),
        ([-129, 32767], "int16"),
        ([-32769, 2**31 - 1], "int32"),
        ([-(2**31) - 1, 2**63 - 1], "int64"),
    ],
)
def test_integer_storage_preserves_width_boundaries(values: list[int], dtype: str) -> None:
    source = np.array(values, dtype=np.int64)
    compact = compact_integer_array(source)
    assert compact.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(compact, source)
    assert compact.nbytes <= source.nbytes


def test_empty_integer_storage_preserves_the_array() -> None:
    source = np.empty(0, dtype=np.int64)
    assert compact_integer_array(source) is source


def test_compact_offsets_add_sizes_without_unsigned_overflow() -> None:
    index = parse_mp4_index("boundary.mp4", _minimal_mp4([65520, 65525, 65530]))
    assert index.sample_offsets.dtype == np.uint16
    span = index.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    assert span.byte_offset == 65520
    assert span.byte_length == 20
    mini = synthesize_mp4(index, span, b"x" * span.byte_length)
    assert len(mini) == synthesized_mp4_size(index, span)
    round_trip = parse_mp4_index("mini.mp4", mini)
    np.testing.assert_array_equal(round_trip.sample_offsets - round_trip.mdat_payload_offset, [0, 5, 10])


def test_compact_offsets_preserve_co64_and_reordered_chunks() -> None:
    offsets = [2**32 - 5, 2**32 + 40, 2**32 + 15]
    index = parse_mp4_index("large.mp4", _minimal_mp4(offsets, use_co64=True))
    np.testing.assert_array_equal(index.sample_offsets, offsets)
    assert index.sample_offsets.dtype == np.int64
    span = index.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    assert span.byte_offset == offsets[0]
    assert span.byte_length == 55


def test_compact_sync_samples_do_not_wrap_one_based_indices() -> None:
    prototype = parse_mp4_index("prototype.mp4", _minimal_mp4([10000, 10010, 10020]))
    index = replace(
        prototype,
        sample_pts=np.arange(256, dtype=np.float64),
        sample_durations=np.full(256, 1, dtype=np.uint8),
        sample_composition_offsets=np.zeros(256, dtype=np.uint8),
        sample_sizes=np.ones(256, dtype=np.uint8),
        sample_offsets=np.arange(10000, 10256, dtype=np.uint16),
        sync_samples=np.array([0, 255], dtype=np.uint8),
    )
    span = index.sample_slice(0, 255, keyframe_pad_s=0, keyframe_pad_fraction=0)
    mini = synthesize_mp4(index, span, b"x" * span.byte_length)
    assert len(mini) == synthesized_mp4_size(index, span)
    round_trip = parse_mp4_index("mini.mp4", mini)
    np.testing.assert_array_equal(round_trip.sync_samples, [0, 255])


def test_compact_integer_slices_preserve_negative_composition_times() -> None:
    prototype = parse_mp4_index("prototype.mp4", _minimal_mp4([10000, 10010, 10020]))
    index = replace(
        prototype,
        sample_pts=np.array([0.0, 0.5, 1.0]),
        sample_durations=np.array([250, 250, 250], dtype=np.uint8),
        sample_composition_offsets=np.array([-128, 127, -128], dtype=np.int8),
        sync_samples=np.array([0, 1, 2], dtype=np.uint8),
    )
    wide = replace(
        index,
        sample_durations=index.sample_durations.astype(np.int64),
        sample_composition_offsets=index.sample_composition_offsets.astype(np.int64),
        sync_samples=index.sync_samples.astype(np.int64),
    )
    span = index.sample_slice(0, 1, keyframe_pad_s=0, keyframe_pad_fraction=0)
    payload = b"x" * span.byte_length
    assert synthesize_mp4(index, span, payload) == synthesize_mp4(wide, span, payload)
    assert synthesized_mp4_size(index, span) == synthesized_mp4_size(wide, span)


def test_sidecar_keeps_compact_arrays_and_exact_float_timestamps(tmp_path: Path) -> None:
    index = parse_mp4_index("test.mp4", _minimal_mp4([10000, 10050, 10025]))
    index = replace(index, sample_pts=np.array([0.0, np.nextafter(0.1, 1.0), -0.0]))
    record = VideoFileRecord("test.mp4", 10128, index)
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((record.file_path, record.file_size),))
    output = tmp_path / "compact.npz"
    EpisodeVideoManifest.save_file_sidecar(output, [record], spec=spec)
    assert EpisodeVideoManifest.validate_file_sidecar(output, spec)
    loaded = EpisodeVideoManifest.load_file_sidecar(output)[record.file_path].mp4
    assert loaded.sample_pts.tobytes() == index.sample_pts.tobytes()
    for name in (
        "sample_durations",
        "sample_composition_offsets",
        "sample_sizes",
        "sample_offsets",
        "sync_samples",
    ):
        actual, expected = getattr(loaded, name), getattr(index, name)
        assert actual.dtype == expected.dtype
        assert not actual.flags.writeable
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("entry_point", ["save", "build"])
def test_schema_three_archive_keeps_old_reader_integer_contract(tmp_path: Path, entry_point: str) -> None:
    data = _minimal_mp4([10000, 10010, 10020])
    source = tmp_path / "test.mp4"
    source.write_bytes(data)
    index = parse_mp4_index(source.name, data)
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((source.name, len(data)),))
    output = tmp_path / "index.npz"
    if entry_point == "save":
        EpisodeVideoManifest.save_file_sidecar(
            output, [VideoFileRecord(source.name, len(data), index)], spec=spec
        )
    else:
        EpisodeVideoManifest.write_file_sidecar(
            output, [source.name], tmp_path, spec=spec, range_backend="fsspec", workers=2
        )
    with np.load(output, allow_pickle=False) as archive:
        assert json.loads(bytes(archive["manifest_json"]))["version"] == 3
        for name in sidecar_utils.ARRAY_NAMES:
            array = archive[f"0/{name}"]
            assert array.dtype == (np.float64 if name == "sample_pts" else np.int64)
            np.testing.assert_array_equal(array, getattr(index, name))


def test_canonical_writer_releases_each_widened_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = parse_mp4_index("test.mp4", _minimal_mp4([10000, 10010, 10020]))
    record = VideoFileRecord(index.file_path, index.file_size, index)
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((record.file_path, record.file_size),))
    original = np.lib.format.write_array
    references = []
    integer_names = sidecar_utils.ARRAY_NAMES[1:]
    original_ids = {id(getattr(index, name)) for name in integer_names}
    expected_copies = sum(getattr(index, name).dtype != np.int64 for name in integer_names)

    def write(member: Any, array: np.ndarray, *args: Any, **kwargs: Any) -> None:
        assert all(ref() is None for ref in references), "The writer retains a previous widened array"
        if array.dtype == np.int64 and id(array) not in original_ids:
            references.append(weakref.ref(array))
        original(member, array, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "write_array", write)
    EpisodeVideoManifest.save_file_sidecar(tmp_path / "index.npz", [record], spec=spec)
    assert len(references) == expected_copies
    assert all(ref() is None for ref in references)


def test_sidecar_validation_rejects_float32_timestamps_before_cache_conversion(tmp_path: Path) -> None:
    index = parse_mp4_index("test.mp4", _minimal_mp4([10000, 10010, 10020]))
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    path = tmp_path / "index.npz"
    EpisodeVideoManifest.save_file_sidecar(
        path, [VideoFileRecord(index.file_path, index.file_size, index)], spec=spec
    )
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files}
    members["0/sample_pts"] = members["0/sample_pts"].astype(np.float32)
    np.savez_compressed(path, **members)

    assert not EpisodeVideoManifest.validate_file_sidecar(path, spec, prepare_cache=False)
    assert not EpisodeVideoManifest.validate_file_sidecar(path, spec)
    assert not list((tmp_path / "cache").glob("**/*.bin"))


@pytest.mark.parametrize("invalid", [np.array([2**63] * 3, dtype=np.uint64), np.array([0.5] * 3)])
def test_canonical_writer_rejects_lossy_conversion_and_preserves_output(
    tmp_path: Path, invalid: np.ndarray
) -> None:
    index = parse_mp4_index("test.mp4", _minimal_mp4([10000, 10010, 10020]))
    record = VideoFileRecord(index.file_path, index.file_size, index)
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((record.file_path, record.file_size),))
    path = tmp_path / "index.npz"
    EpisodeVideoManifest.save_file_sidecar(path, [record], spec=spec)
    before = path.read_bytes()
    invalid_record = replace(record, mp4=replace(index, sample_offsets=invalid))
    with pytest.raises(ValueError, match="integers|exceeds int64"):
        EpisodeVideoManifest.save_file_sidecar(path, [invalid_record], spec=spec)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_new_mapping_compacts_legacy_archive_and_isolates_old_cache(tmp_path: Path) -> None:
    index = parse_mp4_index("test.mp4", _minimal_mp4([65520, 65525, 65530]))
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    payload = {
        "version": 3,
        "sidecar": spec.to_dict(),
        "files": [{"file_path": index.file_path, "file_size": index.file_size, "mp4": index.to_dict()}],
    }
    arrays = {
        f"0/{name}": getattr(index, name).astype(np.float64 if name == "sample_pts" else np.int64)
        for name in sidecar_utils.ARRAY_NAMES
    }
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, manifest_json=json.dumps(payload).encode(), **arrays)
    new_cache = sidecar_utils._cache_path(path)
    old_cache = new_cache.parent.parent / "v2" / new_cache.name
    old_cache.parent.mkdir(parents=True, exist_ok=True)
    old_cache.write_bytes(b"legacy process cache")
    cache, metadata = sidecar_utils.mapped_sidecar(path)
    assert cache == new_cache
    assert cache.parent.name == "v3"
    assert old_cache.parent.name == "v2"
    assert cache != old_cache
    assert cache.read_bytes()[-8:] == b"LRIDX003"
    assert metadata["cache_version"] == 3
    assert old_cache.read_bytes() == b"legacy process cache"
    loaded = EpisodeVideoManifest.load_file_sidecar(path)[index.file_path].mp4
    assert loaded.sample_offsets.dtype == np.uint16
    assert loaded.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0).byte_length == 20
    assert loaded.sample_pts.tobytes() == index.sample_pts.tobytes()
    for name in sidecar_utils.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(index, name))


def test_constant_arrays_map_as_read_only_zero_stride_views(tmp_path: Path) -> None:
    source = _minimal_mp4([10000, 10050, 10025])
    index = parse_mp4_index("constant.mp4", source)
    index = replace(
        index,
        sample_composition_offsets=np.full(3, -7, dtype=np.int64),
        sync_samples=np.empty(0, dtype=np.int64),
    )
    spec = SidecarSpec("test/constant", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    path = tmp_path / "constant.npz"
    EpisodeVideoManifest.save_file_sidecar(
        path, [VideoFileRecord(index.file_path, index.file_size, index)], spec=spec
    )

    cache, payload = sidecar_utils.mapped_sidecar(path)
    item = payload["files"][0]
    buffer = np.memmap(cache, mode="r", dtype=np.uint8)
    arrays = sidecar_utils.mapped_arrays(buffer, item)

    for name in ("sample_durations", "sample_composition_offsets", "sample_sizes"):
        array = arrays[name]
        assert array.shape == (3,)
        assert array.strides == (0,)
        assert not array.flags.writeable
        np.testing.assert_array_equal(array, getattr(index, name))
    assert arrays["sync_samples"].shape == (0,)
    assert arrays["sync_samples"].strides != (0,)
    assert arrays["sample_offsets"].strides != (0,)
    assert item["arrays"]["sample_composition_offsets"][3] == 1
    assert item["arrays"]["sample_pts"][3] == 3
    assert arrays["sample_composition_offsets"].base is buffer
    del buffer
    assert arrays["sample_composition_offsets"].base is not None
    assert bool(np.any(arrays["sample_composition_offsets"]))


def test_singleton_and_empty_array_mappings_keep_shape_and_values(tmp_path: Path) -> None:
    index = parse_mp4_index("singleton.mp4", _minimal_mp4([10000, 10050, 10025]))
    index = replace(
        index,
        sample_pts=index.sample_pts[:1],
        sample_durations=index.sample_durations[:1],
        sample_composition_offsets=np.array([-9], dtype=np.int64),
        sample_sizes=index.sample_sizes[:1],
        sample_offsets=index.sample_offsets[:1],
        sync_samples=np.empty(0, dtype=np.int64),
    )
    spec = SidecarSpec("test/singleton", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    path = tmp_path / "singleton.npz"
    EpisodeVideoManifest.save_file_sidecar(
        path, [VideoFileRecord(index.file_path, index.file_size, index)], spec=spec
    )
    cache, payload = sidecar_utils.mapped_sidecar(path)
    arrays = sidecar_utils.mapped_arrays(np.memmap(cache, mode="r", dtype=np.uint8), payload["files"][0])
    assert arrays["sample_composition_offsets"].shape == (1,)
    np.testing.assert_array_equal(arrays["sample_composition_offsets"], [-9])
    assert arrays["sync_samples"].shape == (0,)
    assert not arrays["sync_samples"].flags.writeable


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("sample_pts", lambda descriptor: descriptor.append(3)),
        ("sample_pts", lambda descriptor: descriptor.__setitem__(2, "<f4")),
        ("sample_sizes", lambda descriptor: descriptor.__setitem__(3, 2)),
        ("sample_offsets", lambda descriptor: descriptor.__setitem__(3, 4)),
        ("sample_durations", lambda descriptor: descriptor.__setitem__(0, 10**9)),
        ("sync_samples", lambda descriptor: descriptor.__setitem__(2, "|O")),
        ("sync_samples", lambda descriptor: descriptor.__setitem__(1, 2**63)),
        ("sync_samples", lambda descriptor: descriptor.__setitem__(slice(None), [0, 2**62, "<u8", 1])),
    ],
)
def test_invalid_v3_descriptors_are_recreated(tmp_path: Path, field: str, mutate: Any) -> None:
    index = parse_mp4_index("invalid.mp4", _minimal_mp4([10000, 10050, 10025]))
    spec = SidecarSpec("test/invalid", "rev", str(tmp_path), ((index.file_path, index.file_size),))
    path = tmp_path / "invalid.npz"
    EpisodeVideoManifest.save_file_sidecar(
        path, [VideoFileRecord(index.file_path, index.file_size, index)], spec=spec
    )
    cache, payload = sidecar_utils.mapped_sidecar(path)
    raw = cache.read_bytes()
    old_length = int.from_bytes(raw[-16:-8], "little")
    mutate(payload["files"][0]["arrays"][field])
    metadata = json.dumps(payload).encode()
    cache.write_bytes(raw[: -16 - old_length] + metadata + len(metadata).to_bytes(8, "little") + raw[-8:])
    loaded = EpisodeVideoManifest.load_file_sidecar(path)[index.file_path].mp4
    np.testing.assert_array_equal(loaded.sample_offsets, index.sample_offsets)
