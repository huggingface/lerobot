# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared, read-only local representation of compressed MP4 sidecar arrays."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock
from huggingface_hub.constants import HF_HOME

from lerobot.streaming.mp4 import Mp4Index

ARRAY_NAMES = (
    "sample_pts",
    "sample_durations",
    "sample_composition_offsets",
    "sample_sizes",
    "sample_offsets",
    "sync_samples",
)
_MAGIC = b"LRIDX001"


def _signature(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _cache_path(path: Path, signature: tuple[int, ...] | None = None) -> Path:
    signature = _signature(path.stat()) if signature is None else signature
    digest = hashlib.sha256(repr(signature).encode()).hexdigest()[:24]
    cache_root = Path(os.environ.get("HF_LEROBOT_HOME", str(Path(HF_HOME) / "lerobot"))).expanduser()
    return cache_root / "streaming-indexes" / f"mp4-{digest}.bin"


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        source.seek(-16, os.SEEK_END)
        length = int.from_bytes(source.read(8), "little")
        if source.read(8) != _MAGIC or not 0 < length <= size - 16:
            raise ValueError("Invalid mapped MP4 index footer")
        source.seek(size - 16 - length)
        payload = json.loads(source.read(length))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 3
        or not isinstance(payload.get("sidecar"), dict)
        or not isinstance(payload.get("files"), list)
    ):
        raise ValueError("Invalid mapped MP4 index schema")
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("arrays"), dict):
            raise ValueError("Invalid mapped MP4 index record")
        for name in ARRAY_NAMES:
            offset, count, dtype_string = item["arrays"][name]
            dtype = np.dtype(dtype_string)
            if (
                type(offset) is not int
                or type(count) is not int
                or dtype.kind not in "iuf"
                or count < 0
                or offset < 0
                or offset % dtype.alignment
                or offset + count * dtype.itemsize > size - 16 - length
            ):
                raise ValueError("Invalid mapped MP4 index array")
        counts = {item["arrays"][name][1] for name in ARRAY_NAMES if name != "sync_samples"}
        if len(counts) != 1:
            raise ValueError("Inconsistent MP4 sample counts")
    return payload


def sidecar_payload(path: Path) -> dict[str, Any]:
    """Read identity without decompressing arrays or preparing a derived cache."""
    try:
        return _read_metadata(_cache_path(path))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    with np.load(path, allow_pickle=False) as data:
        payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
    if payload.get("version") != 3 or not isinstance(payload.get("sidecar"), dict):
        raise ValueError(f"Unsupported MP4 sidecar schema in {path}")
    return payload


def validate_source_arrays(path: Path, payload: dict[str, Any]) -> None:
    """Check temporary NPZ contents without retaining arrays or creating a cache."""
    with np.load(path, allow_pickle=False) as data:
        for file_index, item in enumerate(payload["files"]):
            arrays = {name: data[f"{file_index}/{name}"] for name in ARRAY_NAMES}
            _validate_arrays(arrays)
            Mp4Index.from_dict(item["mp4"], arrays)


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        if array.ndim != 1 or array.dtype.kind not in "iuf":
            raise ValueError(f"Invalid MP4 sample array: {name}")
    if len({len(array) for name, array in arrays.items() if name != "sync_samples"}) != 1:
        raise ValueError("Inconsistent MP4 sample counts")


def mapped_sidecar(path: Path) -> tuple[Path, dict[str, Any]]:
    """Convert once under a lock, then reuse an immutable memory-mappable index.

    Only index metadata is cached, never video payloads. Each source generation gets
    a different file: replacing a sidecar cannot invalidate live array views.
    """
    signature = _signature(path.stat())
    destination = _cache_path(path, signature)

    def cached() -> dict[str, Any] | None:
        try:
            return _read_metadata(destination)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    payload = cached()
    if payload is not None:
        return destination, payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock", timeout=30 * 60):
        payload = cached()
        if payload is not None:
            return destination, payload
        temporary: Path | None = None
        try:
            with path.open("rb") as source, np.load(source, allow_pickle=False) as data:
                if _signature(os.fstat(source.fileno())) != signature:
                    raise OSError("MP4 sidecar changed before index conversion")
                payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
                if payload.get("version") != 3 or not isinstance(payload.get("sidecar"), dict):
                    raise ValueError(f"Unsupported MP4 sidecar schema in {path}")
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".index.tmp", delete=False
                ) as out:
                    temporary = Path(out.name)
                    for file_index, item in enumerate(payload["files"]):
                        arrays = {name: data[f"{file_index}/{name}"] for name in ARRAY_NAMES}
                        _validate_arrays(arrays)
                        Mp4Index.from_dict(item["mp4"], arrays)
                        item["arrays"] = {}
                        for name, array in arrays.items():
                            out.write(b"\0" * (-out.tell() % 8))
                            item["arrays"][name] = [out.tell(), array.size, array.dtype.str]
                            out.write(array.tobytes())
                    if _signature(os.fstat(source.fileno())) != signature:
                        raise OSError("MP4 sidecar changed during index conversion")
                    metadata = json.dumps(payload, separators=(",", ":")).encode()
                    out.write(metadata)
                    out.write(len(metadata).to_bytes(8, "little"))
                    out.write(_MAGIC)
                    out.flush()
                    os.fsync(out.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination, payload


def mapped_arrays(buffer: np.memmap, item: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return views whose base retains the single shared read-only mapping."""
    return {
        name: np.ndarray((count,), dtype=np.dtype(dtype), buffer=buffer, offset=offset)
        for name, (offset, count, dtype) in item["arrays"].items()
    }
