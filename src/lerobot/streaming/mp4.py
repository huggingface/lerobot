# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MP4 indexing and in-memory episode synthesis primitives."""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    type: bytes
    start: int
    header_size: int
    end: int

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Mp4SampleSlice:
    sample_lo: int
    sample_hi: int
    byte_offset: int
    byte_length: int
    source_start_pts: float


@dataclass(frozen=True)
class Mp4Index:
    file_path: str
    file_size: int
    ftyp: bytes
    moov_offset: int
    mdat_offset: int
    mdat_payload_offset: int
    mdat_payload_size: int
    faststart: bool
    codec: str
    timescale: int
    duration: int
    track_id: int
    width: int
    height: int
    stsd_body: bytes
    sample_pts: np.ndarray
    sample_durations: np.ndarray
    sample_sizes: np.ndarray
    sample_offsets: np.ndarray
    sync_samples: np.ndarray

    def sample_slice(
        self,
        from_ts: float,
        to_ts: float,
        *,
        keyframe_pad_s: float = 0.1,
        keyframe_pad_fraction: float = 0.05,
        file_size: int | None = None,
    ) -> Mp4SampleSlice:
        if to_ts < from_ts:
            raise ValueError(f"Invalid timestamp span: {from_ts=} {to_ts=}")
        if len(self.sample_pts) == 0:
            raise ValueError(f"{self.file_path} contains no indexed samples")

        pad = max(keyframe_pad_s, (to_ts - from_ts) * keyframe_pad_fraction)
        lo_ts = max(0.0, from_ts - pad)
        hi_ts = to_ts + pad
        lo = int(np.searchsorted(self.sample_pts, lo_ts, side="left"))
        hi = int(np.searchsorted(self.sample_pts, hi_ts, side="right")) - 1
        lo = min(max(lo, 0), len(self.sample_pts) - 1)
        hi = min(max(hi, lo), len(self.sample_pts) - 1)

        if len(self.sync_samples):
            prev_sync = self.sync_samples[self.sync_samples <= lo]
            if len(prev_sync):
                lo = int(prev_sync[-1])
            else:
                lo = int(self.sync_samples[0])
                if lo > hi:
                    hi = lo

        offsets = self.sample_offsets[lo : hi + 1]
        sizes = self.sample_sizes[lo : hi + 1]
        slice_lo = int(offsets.min())
        slice_hi = int((offsets + sizes).max())
        if file_size is not None:
            slice_hi = min(slice_hi, int(file_size))
        return Mp4SampleSlice(
            sample_lo=lo,
            sample_hi=hi,
            byte_offset=slice_lo,
            byte_length=slice_hi - slice_lo,
            source_start_pts=float(self.sample_pts[lo]),
        )

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_size": self.file_size,
            "ftyp": self.ftyp.hex(),
            "moov_offset": self.moov_offset,
            "mdat_offset": self.mdat_offset,
            "mdat_payload_offset": self.mdat_payload_offset,
            "mdat_payload_size": self.mdat_payload_size,
            "faststart": self.faststart,
            "codec": self.codec,
            "timescale": self.timescale,
            "duration": self.duration,
            "track_id": self.track_id,
            "width": self.width,
            "height": self.height,
            "stsd_body": self.stsd_body.hex(),
        }

    @classmethod
    def from_dict(cls, data: dict, arrays: dict[str, np.ndarray]) -> Mp4Index:
        return cls(
            file_path=data["file_path"],
            file_size=int(data["file_size"]),
            ftyp=bytes.fromhex(data["ftyp"]),
            moov_offset=int(data["moov_offset"]),
            mdat_offset=int(data["mdat_offset"]),
            mdat_payload_offset=int(data["mdat_payload_offset"]),
            mdat_payload_size=int(data["mdat_payload_size"]),
            faststart=bool(data["faststart"]),
            codec=data["codec"],
            timescale=int(data["timescale"]),
            duration=int(data["duration"]),
            track_id=int(data["track_id"]),
            width=int(data["width"]),
            height=int(data["height"]),
            stsd_body=bytes.fromhex(data["stsd_body"]),
            sample_pts=arrays["sample_pts"],
            sample_durations=arrays["sample_durations"],
            sample_sizes=arrays["sample_sizes"],
            sample_offsets=arrays["sample_offsets"],
            sync_samples=arrays["sync_samples"],
        )


def fetch_mp4_index(
    path: str,
    read_range: Callable[[str, int, int], bytes],
    *,
    file_size: int,
    header_probe_bytes: int = 4 * 1024 * 1024,
    max_probe_bytes: int = 64 * 1024 * 1024,
) -> Mp4Index:
    probe_size = min(header_probe_bytes, file_size)
    while True:
        data = read_range(path, 0, probe_size)
        top = list(iter_boxes(data, 0, len(data), absolute_base=0, allow_truncated=True))
        has_mdat = any(box.type == b"mdat" for box in top)
        has_moov = any(box.type == b"moov" and box.end <= len(data) for box in top)
        if has_mdat and has_moov:
            return parse_mp4_index(path, data, file_size=file_size)
        if probe_size >= min(max_probe_bytes, file_size):
            if has_mdat and not has_moov:
                tail_index = _fetch_tail_moov_index(path, read_range, data, top, file_size, max_probe_bytes)
                if tail_index is not None:
                    return tail_index
            missing = []
            if not has_mdat:
                missing.append("mdat")
            if not has_moov:
                missing.append("moov")
            raise ValueError(
                f"Could not find complete {'/'.join(missing)} in first {probe_size} bytes of {path}"
            )
        probe_size = min(probe_size * 2, max_probe_bytes, file_size)


def _fetch_tail_moov_index(
    path: str,
    read_range: Callable[[str, int, int], bytes],
    prefix: bytes,
    top_boxes: list[Box],
    file_size: int,
    max_probe_bytes: int,
) -> Mp4Index | None:
    mdat_box = _one(top_boxes, b"mdat")
    if mdat_box is None or mdat_box.end >= file_size:
        return None
    tail_offset = mdat_box.end
    tail_length = min(max_probe_bytes, file_size - tail_offset)
    tail = read_range(path, tail_offset, tail_length)
    tail_boxes = list(iter_boxes(tail, 0, len(tail), absolute_base=tail_offset, allow_truncated=True))
    moov_box = next(
        (box for box in tail_boxes if box.type == b"moov" and box.end <= tail_offset + len(tail)), None
    )
    if moov_box is None:
        return None
    ftyp_box = _one(top_boxes, b"ftyp", required=False)
    ftyp = (
        prefix[ftyp_box.start : ftyp_box.end]
        if ftyp_box is not None
        else _box(b"ftyp", b"isom\0\0\2\0isomiso2mp41")
    )
    moov_start = moov_box.payload_start - tail_offset
    moov_end = moov_box.end - tail_offset
    return _parse_mp4_index_from_layout(
        path,
        file_size=file_size,
        ftyp=ftyp,
        moov_offset=moov_box.start,
        moov=tail[moov_start:moov_end],
        mdat_box=mdat_box,
    )


def parse_mp4_index(path: str, data: bytes, *, file_size: int | None = None) -> Mp4Index:
    if file_size is None:
        file_size = len(data)
    top = list(iter_boxes(data, 0, len(data), absolute_base=0, allow_truncated=True))
    ftyp_box = _one(top, b"ftyp", required=False)
    moov_box = _one(top, b"moov")
    mdat_box = _one(top, b"mdat")
    if moov_box.end > len(data):
        raise ValueError(f"{path}: moov box is truncated")

    moov = data[moov_box.payload_start : moov_box.end]
    ftyp = (
        data[ftyp_box.start : ftyp_box.end]
        if ftyp_box is not None
        else _box(b"ftyp", b"isom\0\0\2\0isomiso2mp41")
    )
    return _parse_mp4_index_from_layout(
        path,
        file_size=file_size,
        ftyp=ftyp,
        moov_offset=moov_box.start,
        moov=moov,
        mdat_box=mdat_box,
    )


def _parse_mp4_index_from_layout(
    path: str,
    *,
    file_size: int,
    ftyp: bytes,
    moov_offset: int,
    moov: bytes,
    mdat_box: Box,
) -> Mp4Index:
    mvhd_timescale, mvhd_duration = _parse_mvhd(_find_descendant(moov, [b"mvhd"]))
    trak_box, trak_payload = _find_video_trak(moov)
    _ = trak_box
    tkhd = _parse_tkhd(_find_descendant(trak_payload, [b"tkhd"]))
    mdhd_timescale, mdhd_duration = _parse_mdhd(_find_descendant(trak_payload, [b"mdia", b"mdhd"]))
    stbl = _find_descendant(trak_payload, [b"mdia", b"minf", b"stbl"])

    stsd = _find_child(stbl, b"stsd")
    stsd_body = stbl[stsd.payload_start : stsd.end]
    codec = _parse_stsd_codec(stsd_body)
    stts = _parse_stts(_payload(stbl, b"stts"))
    sample_sizes = _parse_stsz(_payload(stbl, b"stsz"))
    stsc = _parse_stsc(_payload(stbl, b"stsc"))
    chunk_offsets = _parse_chunk_offsets(stbl)
    sync_samples = _parse_stss(stbl, len(sample_sizes))

    sample_durations = _expand_stts(stts, len(sample_sizes))
    sample_pts_units = np.empty(len(sample_durations), dtype=np.int64)
    if len(sample_durations):
        sample_pts_units[0] = 0
        if len(sample_durations) > 1:
            sample_pts_units[1:] = np.cumsum(sample_durations[:-1], dtype=np.int64)
    sample_pts = sample_pts_units.astype(np.float64) / float(mdhd_timescale)
    sample_offsets = _sample_offsets(stsc, chunk_offsets, sample_sizes)

    return Mp4Index(
        file_path=path,
        file_size=file_size,
        ftyp=ftyp,
        moov_offset=moov_offset,
        mdat_offset=mdat_box.start,
        mdat_payload_offset=mdat_box.payload_start,
        mdat_payload_size=mdat_box.end - mdat_box.payload_start
        if mdat_box.end <= file_size
        else file_size - mdat_box.payload_start,
        faststart=moov_offset < mdat_box.start,
        codec=codec,
        timescale=mdhd_timescale,
        duration=mdhd_duration or mvhd_duration,
        track_id=tkhd["track_id"],
        width=tkhd["width"],
        height=tkhd["height"],
        stsd_body=stsd_body,
        sample_pts=sample_pts,
        sample_durations=sample_durations,
        sample_sizes=sample_sizes,
        sample_offsets=sample_offsets,
        sync_samples=sync_samples,
    )


def synthesize_mp4(index: Mp4Index, sample_slice: Mp4SampleSlice, mdat_payload: bytes) -> bytes:
    lo = sample_slice.sample_lo
    hi = sample_slice.sample_hi + 1
    if lo < 0 or hi > len(index.sample_sizes) or lo >= hi:
        raise ValueError(f"Invalid sample range [{lo}, {hi}) for {index.file_path}")

    offsets = index.sample_offsets[lo:hi]
    sizes = index.sample_sizes[lo:hi]
    rel_offsets = offsets - sample_slice.byte_offset
    if int(rel_offsets.min()) != 0:
        raise ValueError("Sample slice must start at the minimum referenced sample offset")
    if int((rel_offsets + sizes).max()) > len(mdat_payload):
        raise ValueError("Sample slice does not cover all referenced samples")

    durations = index.sample_durations[lo:hi]
    sync = index.sync_samples[(index.sync_samples >= lo) & (index.sync_samples < hi)] - lo + 1
    moov = _make_moov(index, durations, sizes, rel_offsets, sync, mdat_data_offset=0)
    header_size = len(index.ftyp) + len(moov)
    mdat_header_size = 8 if len(mdat_payload) + 8 <= 0xFFFFFFFF else 16
    moov = _make_moov(
        index,
        durations,
        sizes,
        rel_offsets,
        sync,
        mdat_data_offset=header_size + mdat_header_size,
    )
    return index.ftyp + moov + _box(b"mdat", mdat_payload)


def synthesized_mp4_size(index: Mp4Index, sample_slice: Mp4SampleSlice) -> int:
    """Return the exact synthesized mini-MP4 size without fetching its media payload."""
    lo = sample_slice.sample_lo
    hi = sample_slice.sample_hi + 1
    if lo < 0 or hi > len(index.sample_sizes) or lo >= hi:
        raise ValueError(f"Invalid sample range [{lo}, {hi}) for {index.file_path}")

    offsets = index.sample_offsets[lo:hi]
    sizes = index.sample_sizes[lo:hi]
    rel_offsets = offsets - sample_slice.byte_offset
    if int(rel_offsets.min()) != 0:
        raise ValueError("Sample slice must start at the minimum referenced sample offset")
    if int((rel_offsets + sizes).max()) > sample_slice.byte_length:
        raise ValueError("Sample slice does not cover all referenced samples")

    durations = index.sample_durations[lo:hi]
    sync = index.sync_samples[(index.sync_samples >= lo) & (index.sync_samples < hi)] - lo + 1
    moov = _make_moov(index, durations, sizes, rel_offsets, sync, mdat_data_offset=0)
    header_size = len(index.ftyp) + len(moov)
    mdat_header_size = 8 if sample_slice.byte_length + 8 <= 0xFFFFFFFF else 16
    moov = _make_moov(
        index,
        durations,
        sizes,
        rel_offsets,
        sync,
        mdat_data_offset=header_size + mdat_header_size,
    )
    return len(index.ftyp) + len(moov) + mdat_header_size + sample_slice.byte_length


def iter_boxes(
    data: bytes,
    start: int,
    end: int,
    *,
    absolute_base: int = 0,
    allow_truncated: bool = False,
) -> Iterable[Box]:
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        typ = data[pos + 4 : pos + 8]
        header_size = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - pos
        if size < header_size:
            break
        box_end = pos + size
        if box_end > end and not allow_truncated:
            break
        yield Box(typ, absolute_base + pos, header_size, absolute_base + box_end)
        pos = box_end


def _find_video_trak(moov: bytes) -> tuple[Box, bytes]:
    for trak in _children(moov, 0, len(moov)):
        if trak.type != b"trak":
            continue
        payload = moov[trak.payload_start : trak.end]
        hdlr = _find_descendant(payload, [b"mdia", b"hdlr"])
        if hdlr[8:12] == b"vide":
            return trak, payload
    raise ValueError("No video track found")


def _find_descendant(data: bytes, path: list[bytes]) -> bytes:
    current = data
    for typ in path:
        box = _find_child(current, typ)
        current = current[box.payload_start : box.end]
    return current


def _find_child(data: bytes, typ: bytes) -> Box:
    for box in _children(data, 0, len(data)):
        if box.type == typ:
            return box
    raise ValueError(f"Missing MP4 box {typ.decode('latin1')}")


def _children(data: bytes, start: int, end: int) -> Iterable[Box]:
    return iter_boxes(data, start, end, absolute_base=0)


def _one(boxes: list[Box], typ: bytes, *, required: bool = True) -> Box | None:
    matches = [box for box in boxes if box.type == typ]
    if not matches and required:
        raise ValueError(f"Missing MP4 box {typ.decode('latin1')}")
    return matches[0] if matches else None


def _payload(parent: bytes, typ: bytes) -> bytes:
    box = _find_child(parent, typ)
    return parent[box.payload_start : box.end]


def _parse_mvhd(payload: bytes) -> tuple[int, int]:
    version = payload[0]
    if version == 1:
        return struct.unpack_from(">IQ", payload, 20)
    return struct.unpack_from(">II", payload, 12)


def _parse_mdhd(payload: bytes) -> tuple[int, int]:
    version = payload[0]
    if version == 1:
        return struct.unpack_from(">IQ", payload, 20)
    return struct.unpack_from(">II", payload, 12)


def _parse_tkhd(payload: bytes) -> dict[str, int]:
    version = payload[0]
    if version == 1:
        track_id = struct.unpack_from(">I", payload, 20)[0]
        duration = struct.unpack_from(">Q", payload, 28)[0]
        width, height = struct.unpack_from(">II", payload, 88)
    else:
        track_id = struct.unpack_from(">I", payload, 12)[0]
        duration = struct.unpack_from(">I", payload, 20)[0]
        width, height = struct.unpack_from(">II", payload, 76)
    return {"track_id": track_id, "duration": duration, "width": width >> 16, "height": height >> 16}


def _parse_stsd_codec(stsd_body: bytes) -> str:
    if len(stsd_body) < 16:
        return "unknown"
    return stsd_body[12:16].decode("latin1")


def _parse_stts(payload: bytes) -> list[tuple[int, int]]:
    count = struct.unpack_from(">I", payload, 4)[0]
    out = []
    offset = 8
    for _ in range(count):
        out.append(struct.unpack_from(">II", payload, offset))
        offset += 8
    return out


def _expand_stts(entries: list[tuple[int, int]], sample_count: int) -> np.ndarray:
    values = np.empty(sample_count, dtype=np.int64)
    pos = 0
    for count, delta in entries:
        values[pos : pos + count] = delta
        pos += count
    if pos != sample_count:
        raise ValueError(f"stts describes {pos} samples, stsz describes {sample_count}")
    return values


def _parse_stsz(payload: bytes) -> np.ndarray:
    sample_size, sample_count = struct.unpack_from(">II", payload, 4)
    if sample_size:
        return np.full(sample_count, sample_size, dtype=np.int64)
    offset = 12
    values = np.empty(sample_count, dtype=np.int64)
    for idx in range(sample_count):
        values[idx] = struct.unpack_from(">I", payload, offset)[0]
        offset += 4
    return values


def _parse_stsc(payload: bytes) -> list[tuple[int, int, int]]:
    count = struct.unpack_from(">I", payload, 4)[0]
    out = []
    offset = 8
    for _ in range(count):
        out.append(struct.unpack_from(">III", payload, offset))
        offset += 12
    return out


def _parse_chunk_offsets(stbl: bytes) -> np.ndarray:
    with_stco = None
    with_co64 = None
    for box in _children(stbl, 0, len(stbl)):
        if box.type == b"stco":
            with_stco = stbl[box.payload_start : box.end]
        elif box.type == b"co64":
            with_co64 = stbl[box.payload_start : box.end]
    if with_co64 is not None:
        count = struct.unpack_from(">I", with_co64, 4)[0]
        return np.array(
            [struct.unpack_from(">Q", with_co64, 8 + idx * 8)[0] for idx in range(count)], dtype=np.int64
        )
    if with_stco is None:
        raise ValueError("Missing stco/co64 chunk offsets")
    count = struct.unpack_from(">I", with_stco, 4)[0]
    return np.array(
        [struct.unpack_from(">I", with_stco, 8 + idx * 4)[0] for idx in range(count)], dtype=np.int64
    )


def _parse_stss(stbl: bytes, sample_count: int) -> np.ndarray:
    for box in _children(stbl, 0, len(stbl)):
        if box.type == b"stss":
            payload = stbl[box.payload_start : box.end]
            count = struct.unpack_from(">I", payload, 4)[0]
            return np.array(
                [struct.unpack_from(">I", payload, 8 + idx * 4)[0] - 1 for idx in range(count)],
                dtype=np.int64,
            )
    return np.arange(sample_count, dtype=np.int64)


def _sample_offsets(
    stsc: list[tuple[int, int, int]], chunk_offsets: np.ndarray, sample_sizes: np.ndarray
) -> np.ndarray:
    if not stsc:
        raise ValueError("stsc is empty")
    offsets = np.empty(len(sample_sizes), dtype=np.int64)
    sample_idx = 0
    for entry_idx, (first_chunk, samples_per_chunk, _desc_idx) in enumerate(stsc):
        next_first = stsc[entry_idx + 1][0] if entry_idx + 1 < len(stsc) else len(chunk_offsets) + 1
        for chunk_number in range(first_chunk, next_first):
            if chunk_number < 1 or chunk_number > len(chunk_offsets):
                raise ValueError("stsc references a chunk outside stco/co64")
            chunk_pos = int(chunk_offsets[chunk_number - 1])
            for _ in range(samples_per_chunk):
                if sample_idx >= len(sample_sizes):
                    return offsets
                offsets[sample_idx] = chunk_pos
                chunk_pos += int(sample_sizes[sample_idx])
                sample_idx += 1
    if sample_idx != len(sample_sizes):
        raise ValueError(f"stsc describes {sample_idx} samples, stsz describes {len(sample_sizes)}")
    return offsets


def _make_moov(
    index: Mp4Index,
    durations: np.ndarray,
    sizes: np.ndarray,
    rel_offsets: np.ndarray,
    sync_samples: np.ndarray,
    *,
    mdat_data_offset: int,
) -> bytes:
    duration = int(durations.sum())
    stco_values = [int(mdat_data_offset + value) for value in rel_offsets]
    if any(value > 0xFFFFFFFF for value in stco_values):
        offset_box = _co64(stco_values)
    else:
        offset_box = _stco(stco_values)
    stbl = _box(
        b"stbl",
        _box(b"stsd", index.stsd_body)
        + _stts(durations)
        + _stsc_one_sample_per_chunk(len(sizes))
        + _stsz(sizes)
        + offset_box
        + (_stss(sync_samples) if len(sync_samples) else b""),
    )
    minf = _box(b"minf", _vmhd() + _dinf() + stbl)
    mdia = _box(b"mdia", _mdhd(index.timescale, duration) + _hdlr() + minf)
    trak = _box(b"trak", _tkhd(index.track_id, duration, index.width, index.height) + mdia)
    return _box(b"moov", _mvhd(index.timescale, duration, index.track_id + 1) + trak)


def _full_box(typ: bytes, version: int, flags: int, payload: bytes = b"") -> bytes:
    return _box(typ, bytes([version]) + flags.to_bytes(3, "big") + payload)


def _box(typ: bytes, payload: bytes) -> bytes:
    size = len(payload) + 8
    if size <= 0xFFFFFFFF:
        return struct.pack(">I4s", size, typ) + payload
    return struct.pack(">I4sQ", 1, typ, size + 8) + payload


def _mvhd(timescale: int, duration: int, next_track_id: int) -> bytes:
    matrix = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    payload = (
        struct.pack(">IIII", 0, 0, timescale, duration)
        + struct.pack(">IHH", 0x00010000, 0x0100, 0)
        + b"\0" * 8
        + matrix
        + b"\0" * 24
        + struct.pack(">I", next_track_id)
    )
    return _full_box(b"mvhd", 0, 0, payload)


def _tkhd(track_id: int, duration: int, width: int, height: int) -> bytes:
    matrix = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    payload = (
        struct.pack(">IIIII", 0, 0, track_id, 0, duration)
        + b"\0" * 8
        + struct.pack(">hhhh", 0, 0, 0, 0)
        + matrix
        + struct.pack(">II", width << 16, height << 16)
    )
    return _full_box(b"tkhd", 0, 7, payload)


def _mdhd(timescale: int, duration: int) -> bytes:
    return _full_box(b"mdhd", 0, 0, struct.pack(">IIIIH", 0, 0, timescale, duration, 0x55C4) + b"\0\0")


def _hdlr() -> bytes:
    return _full_box(b"hdlr", 0, 0, b"\0" * 4 + b"vide" + b"\0" * 12 + b"VideoHandler\0")


def _vmhd() -> bytes:
    return _full_box(b"vmhd", 0, 1, struct.pack(">HHHH", 0, 0, 0, 0))


def _dinf() -> bytes:
    url = _full_box(b"url ", 0, 1)
    dref = _full_box(b"dref", 0, 0, struct.pack(">I", 1) + url)
    return _box(b"dinf", dref)


def _stts(durations: np.ndarray) -> bytes:
    runs = []
    for duration in durations.tolist():
        if runs and runs[-1][1] == int(duration):
            runs[-1][0] += 1
        else:
            runs.append([1, int(duration)])
    payload = struct.pack(">I", len(runs)) + b"".join(
        struct.pack(">II", count, delta) for count, delta in runs
    )
    return _full_box(b"stts", 0, 0, payload)


def _stsc_one_sample_per_chunk(sample_count: int) -> bytes:
    return _full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))


def _stsz(sizes: np.ndarray) -> bytes:
    return _full_box(
        b"stsz",
        0,
        0,
        struct.pack(">II", 0, len(sizes)) + b"".join(struct.pack(">I", int(size)) for size in sizes.tolist()),
    )


def _stco(values: list[int]) -> bytes:
    return _full_box(
        b"stco", 0, 0, struct.pack(">I", len(values)) + b"".join(struct.pack(">I", v) for v in values)
    )


def _co64(values: list[int]) -> bytes:
    return _full_box(
        b"co64", 0, 0, struct.pack(">I", len(values)) + b"".join(struct.pack(">Q", v) for v in values)
    )


def _stss(values: np.ndarray) -> bytes:
    return _full_box(
        b"stss",
        0,
        0,
        struct.pack(">I", len(values)) + b"".join(struct.pack(">I", int(value)) for value in values.tolist()),
    )
