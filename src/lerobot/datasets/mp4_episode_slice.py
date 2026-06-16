"""MP4 moov parsing and tight per-episode mdat byte-range fetching.

LeRobot v3 concatenates episodes into shared MP4 files (faststart: moov at head).
For streaming we fetch only the file header plus the episode's contiguous mdat span
instead of the ``0..episode_end`` prefix.
"""

from __future__ import annotations

import io
import struct
import threading
from dataclasses import dataclass, field
from typing import Callable

KEYFRAME_PAD_S = 0.1
HEADER_PROBE_BYTES = 4 * 1024 * 1024
MAX_HEADER_PROBE_BYTES = 16 * 1024 * 1024


@dataclass
class Mp4FileLayout:
    file_size: int
    moov_offset: int
    moov_length: int
    header_end: int
    mdat_offset: int
    mdat_size: int
    faststart: bool
    codec: str


def parse_mp4_file_layout(header_bytes: bytes, file_size: int) -> Mp4FileLayout:
    """Return top-level MP4 layout (moov/mdat positions, faststart flag)."""
    boxes = list(_iter_boxes(header_bytes))
    moov_offset = mdat_offset = -1
    moov_length = mdat_size = 0
    for off, size, typ, _ in boxes:
        if typ == b"moov" and moov_offset < 0:
            moov_offset, moov_length = off, size
        if typ == b"mdat" and mdat_offset < 0:
            mdat_offset, mdat_size = off, size
    if moov_offset < 0:
        raise ValueError("moov box not found in header probe")
    if mdat_offset < 0:
        raise ValueError("mdat box not found in header probe; increase HEADER_PROBE_BYTES")
    faststart = moov_offset < mdat_offset
    header_end = mdat_offset
    codec = _parse_video_codec(header_bytes)
    return Mp4FileLayout(
        file_size=file_size,
        moov_offset=moov_offset,
        moov_length=moov_length,
        header_end=header_end,
        mdat_offset=mdat_offset,
        mdat_size=mdat_size,
        faststart=faststart,
        codec=codec,
    )


def _parse_video_codec(header_bytes: bytes) -> str:
    moov = _find_box_payload(header_bytes, b"moov")
    if moov is None:
        return "unknown"
    trak = _find_video_trak(moov)
    if trak is None:
        return "unknown"
    stsd = _find_box_payload(_find_box_payload(trak, b"stbl") or b"", b"stsd")
    if stsd is None or len(stsd) < 12:
        return "unknown"
    # stsd: version(1)+flags(3)+entry_count(4)+entry_size(4)+codec(4)
    if len(stsd) >= 12:
        return stsd[8:12].decode("latin1", errors="replace").strip("\x00")
    return "unknown"


def average_fps_from_index(index: Mp4VideoIndex) -> float:
    index.ensure_tables()
    if index.num_samples < 2:
        return 30.0
    duration = index.sample_pts(index.num_samples - 1)
    if duration <= 0:
        return 30.0
    return index.num_samples / duration


def episode_custom_frame_mappings_json(
    index: Mp4VideoIndex, from_ts: float, to_ts: float, keyframe_pad_s: float = KEYFRAME_PAD_S
) -> bytes:
    """Build TorchCodec ``custom_frame_mappings`` JSON for one episode span."""
    import json

    index.ensure_tables()
    lo_idx = _first_sample_at_or_after(index._pts, max(0.0, from_ts - keyframe_pad_s))
    hi_idx = _last_sample_at_or_before(index._pts, to_ts + keyframe_pad_s)
    hi_idx = min(hi_idx, index.num_samples - 1)
    lo_idx = _keyframe_back(index.sync_samples, lo_idx)

    sync = set(index.sync_samples)
    timescale = index.timescale
    # stts deltas for duration per sample (expand stts entries to per-sample delta)
    sample_deltas: list[int] = []
    for count, delta in index.stts:
        sample_deltas.extend([delta] * count)
    while len(sample_deltas) < index.num_samples:
        sample_deltas.append(sample_deltas[-1] if sample_deltas else timescale // 30)

    frames = []
    for idx in range(lo_idx, hi_idx + 1):
        frames.append(
            {
                "pts": int(round(index._pts[idx] * timescale)),
                "duration": int(sample_deltas[idx]),
                "key_frame": int((idx + 1) in sync) if sync else int(idx == lo_idx),
            }
        )
    return json.dumps({"frames": frames}).encode()


def episode_keyframes(
    index: Mp4VideoIndex, from_ts: float, to_ts: float, keyframe_pad_s: float = KEYFRAME_PAD_S
) -> list[tuple[float, int]]:
    """Return (pts_seconds, byte_offset) for sync samples in the episode span."""
    index.ensure_tables()
    span = index.episode_byte_span(from_ts, to_ts, keyframe_pad_s)
    lo_idx = _first_sample_at_or_after(index._pts, max(0.0, from_ts - keyframe_pad_s))
    hi_idx = _last_sample_at_or_before(index._pts, to_ts + keyframe_pad_s)
    if not index.sync_samples:
        return [(index.sample_pts(lo_idx), index.sample_offset(lo_idx))]
    out: list[tuple[float, int]] = []
    for sync_one_based in index.sync_samples:
        idx = sync_one_based - 1
        if lo_idx <= idx <= hi_idx:
            out.append((index.sample_pts(idx), index.sample_offset(idx)))
    return out or [(index.sample_pts(lo_idx), index.sample_offset(lo_idx))]


@dataclass
class EpisodeByteSpan:
    """Absolute file byte ranges to fetch for one episode."""

    file_size: int
    header_end: int
    slice_lo: int
    slice_hi: int

    @property
    def header_bytes(self) -> tuple[int, int]:
        return 0, self.header_end - 1

    @property
    def mdat_bytes(self) -> tuple[int, int]:
        return self.slice_lo, self.slice_hi

    @property
    def total_fetch_bytes(self) -> int:
        header = self.header_end
        mdat = self.slice_hi - self.slice_lo + 1
        return header + mdat


@dataclass
class Mp4VideoIndex:
    file_size: int
    header_end: int
    mdat_offset: int
    mdat_size: int
    timescale: int
    stts: list[tuple[int, int]]
    stsz: list[int]
    stsc: list[tuple[int, int, int]]
    stco: list[int]
    sync_samples: list[int]
    _pts: list[float] = field(default_factory=list, repr=False)
    _offsets: list[int] = field(default_factory=list, repr=False)

    def ensure_tables(self) -> None:
        if self._pts:
            return
        self._pts = _pts_from_stts(self.stts, self.timescale)
        self._offsets = _sample_byte_offsets(self.stsc, self.stco, self.stsz)

    @property
    def num_samples(self) -> int:
        return len(self.stsz)

    def sample_pts(self, index: int) -> float:
        self.ensure_tables()
        return self._pts[index]

    def sample_offset(self, index: int) -> int:
        self.ensure_tables()
        index = max(0, min(index, len(self._offsets) - 1))
        return self._offsets[index]

    def sample_end(self, index: int) -> int:
        return self.sample_offset(index) + self.stsz[index]

    def episode_byte_span(self, from_ts: float, to_ts: float, keyframe_pad_s: float = KEYFRAME_PAD_S) -> EpisodeByteSpan:
        self.ensure_tables()
        n = self.num_samples
        if n == 0:
            raise ValueError("MP4 has no video samples")

        pad = max(keyframe_pad_s, 0.05 * max(0.01, to_ts - from_ts))
        lo_ts = max(0.0, from_ts - pad)
        hi_ts = to_ts + pad

        lo_idx = _first_sample_at_or_after(self._pts, lo_ts)
        hi_idx = _last_sample_at_or_before(self._pts, hi_ts)
        hi_idx = min(hi_idx, n - 1)
        lo_idx = min(lo_idx, n - 1)

        lo_idx = _keyframe_back(self.sync_samples, lo_idx)

        slice_lo = self.sample_offset(lo_idx)
        slice_hi = self.sample_end(min(hi_idx, len(self._offsets) - 1))
        return EpisodeByteSpan(
            file_size=self.file_size,
            header_end=self.header_end,
            slice_lo=slice_lo,
            slice_hi=min(slice_hi, self.file_size - 1),
        )


class SparseMp4Reader(io.BufferedIOBase):
    """Range-backed MP4 reader: header + one mdat span at absolute offsets."""

    def __init__(
        self,
        file_size: int,
        header: bytes,
        mdat_lo: int,
        mdat_bytes: bytes,
        lazy_fetch: Callable[[int, int], bytes] | None = None,
    ):
        self._size = file_size
        self._header = header
        self._mdat_lo = mdat_lo
        self._mdat_hi = mdat_lo + len(mdat_bytes)
        self._mdat = mdat_bytes
        self._lazy_fetch = lazy_fetch
        self._pos = 0
        self._lock = threading.Lock()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._pos
        if size <= 0:
            return b""

        out = bytearray()
        remaining = size
        pos = self._pos
        while remaining > 0 and pos < self._size:
            chunk = self._read_at(pos, remaining)
            if not chunk:
                break
            out.extend(chunk)
            pos += len(chunk)
            remaining -= len(chunk)
        self._pos = pos
        return bytes(out)

    def _read_at(self, pos: int, n: int) -> bytes:
        header_len = len(self._header)
        if pos < header_len:
            end = min(pos + n, header_len)
            return self._header[pos:end]

        if self._mdat_lo <= pos < self._mdat_hi:
            end = min(pos + n, self._mdat_hi)
            off = pos - self._mdat_lo
            return self._mdat[off : off + (end - pos)]

        if self._lazy_fetch is not None:
            with self._lock:
                end = min(pos + n, self._size)
                return self._lazy_fetch(pos, end - 1)

        return b"\x00" * min(n, self._size - pos)


def parse_mp4_index(header_bytes: bytes, file_size: int) -> Mp4VideoIndex:
    """Parse moov sample tables from the file header (faststart layout)."""
    layout = parse_mp4_file_layout(header_bytes, file_size)
    mdat_offset, mdat_size = layout.mdat_offset, layout.mdat_size
    moov = _find_box_payload(header_bytes, b"moov")
    if moov is None:
        raise ValueError("moov box not found in MP4 header probe")

    trak = _find_video_trak(moov)
    if trak is None:
        raise ValueError("video trak not found in moov")

    mdhd = _find_box_payload(trak, b"mdhd")
    if mdhd is None:
        raise ValueError("mdhd not found")
    timescale = _parse_mdhd_timescale(mdhd)

    stbl = _find_box_payload(trak, b"stbl")
    if stbl is None:
        raise ValueError("stbl not found")

    stts = _parse_stts(_find_box_payload(stbl, b"stts"))
    stsz = _parse_stsz(_find_box_payload(stbl, b"stsz"))
    stsc = _parse_stsc(_find_box_payload(stbl, b"stsc"))
    stco_payload = _find_box_payload(stbl, b"stco")
    co64_payload = _find_box_payload(stbl, b"co64")
    if stco_payload is not None:
        stco = _parse_stco(stco_payload)
    elif co64_payload is not None:
        stco = _parse_co64(co64_payload)
    else:
        raise ValueError("stco/co64 not found")

    stss_payload = _find_box_payload(stbl, b"stss")
    sync_samples = _parse_stss(stss_payload) if stss_payload else []

    return Mp4VideoIndex(
        file_size=file_size,
        header_end=layout.header_end,
        mdat_offset=mdat_offset,
        mdat_size=mdat_size,
        timescale=timescale,
        stts=stts,
        stsz=stsz,
        stsc=stsc,
        stco=stco,
        sync_samples=sync_samples,
    )


def _box_header(data: bytes, offset: int) -> tuple[int, bytes, int] | None:
    if offset + 8 > len(data):
        return None
    size, typ = struct.unpack_from(">I4s", data, offset)
    header = 8
    if size == 1:
        if offset + 16 > len(data):
            return None
        size = struct.unpack_from(">Q", data, offset + 8)[0]
        header = 16
    elif size == 0:
        size = len(data) - offset
    return size, typ, header


def _iter_boxes(data: bytes, start: int = 0, end: int | None = None):
    end = end if end is not None else len(data)
    off = start
    while off + 8 <= end:
        hdr = _box_header(data, off)
        if hdr is None or hdr[0] < hdr[2]:
            break
        size, typ, header = hdr
        yield off, size, typ, data[off + header : off + size]
        off += size


def _find_box_payload(data: bytes, target: bytes) -> bytes | None:
    for _, _, typ, payload in _iter_boxes(data):
        if typ == target:
            return payload
        if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
            found = _find_box_payload(payload, target)
            if found is not None:
                return found
    return None


def _find_video_trak(moov: bytes) -> bytes | None:
    for _, _, typ, payload in _iter_boxes(moov):
        if typ != b"trak":
            continue
        hdlr = _find_box_payload(payload, b"hdlr")
        if hdlr is not None and len(hdlr) >= 12 and hdlr[8:12] == b"vide":
            return payload
    return None


def _find_mdat(header_bytes: bytes, file_size: int) -> tuple[int, int]:
    for off, size, typ, _ in _iter_boxes(header_bytes):
        if typ == b"mdat":
            return off, size
    # mdat may start beyond probe; scan from file_size hint unavailable — require probe hit
    raise ValueError("mdat box not found in header probe; increase HEADER_PROBE_BYTES")


def _parse_mdhd_timescale(mdhd: bytes) -> int:
    version = mdhd[0]
    if version == 0:
        return struct.unpack_from(">I", mdhd, 12)[0]
    return struct.unpack_from(">I", mdhd, 20)[0]


def _parse_stts(stts: bytes | None) -> list[tuple[int, int]]:
    if stts is None:
        raise ValueError("stts missing")
    count = struct.unpack_from(">I", stts, 4)[0]
    out = []
    off = 8
    for _ in range(count):
        sample_count, delta = struct.unpack_from(">II", stts, off)
        out.append((sample_count, delta))
        off += 8
    return out


def _parse_stsz(stsz: bytes | None) -> list[int]:
    if stsz is None:
        raise ValueError("stsz missing")
    sample_size, sample_count = struct.unpack_from(">II", stsz, 4)
    if sample_size != 0:
        return [sample_size] * sample_count
    off = 12
    return list(struct.unpack_from(f">{sample_count}I", stsz, off))


def _parse_stsc(stsc: bytes | None) -> list[tuple[int, int, int]]:
    if stsc is None:
        raise ValueError("stsc missing")
    count = struct.unpack_from(">I", stsc, 4)[0]
    out = []
    off = 8
    for _ in range(count):
        first_chunk, samples_per_chunk, sample_desc = struct.unpack_from(">III", stsc, off)
        out.append((first_chunk, samples_per_chunk, sample_desc))
        off += 12
    return out


def _parse_stco(stco: bytes) -> list[int]:
    count = struct.unpack_from(">I", stco, 4)[0]
    return list(struct.unpack_from(f">{count}I", stco, 8))


def _parse_co64(co64: bytes) -> list[int]:
    count = struct.unpack_from(">I", co64, 4)[0]
    return [struct.unpack_from(">Q", co64, 8 + i * 8)[0] for i in range(count)]


def _parse_stss(stss: bytes) -> list[int]:
    count = struct.unpack_from(">I", stss, 4)[0]
    return list(struct.unpack_from(f">{count}I", stss, 8))


def _pts_from_stts(stts: list[tuple[int, int]], timescale: int) -> list[float]:
    pts: list[float] = []
    t = 0
    for count, delta in stts:
        for _ in range(count):
            pts.append(t / timescale)
            t += delta
    return pts


def _sample_byte_offsets(
    stsc: list[tuple[int, int, int]], stco: list[int], stsz: list[int]
) -> list[int]:
    if not stsc:
        stsc = [(1, len(stsz), 1)]

    offsets: list[int] = []
    chunk_idx = 0
    sample_idx = 0
    sc_idx = 0
    num_chunks = len(stco)

    while chunk_idx < num_chunks and sample_idx < len(stsz):
        first_chunk, samples_per_chunk, _ = stsc[min(sc_idx, len(stsc) - 1)]
        if sc_idx + 1 < len(stsc):
            next_first = stsc[sc_idx + 1][0]
            chunks_in_entry = next_first - first_chunk
        else:
            chunks_in_entry = num_chunks - chunk_idx

        for _ in range(chunks_in_entry):
            if chunk_idx >= num_chunks:
                break
            offset = stco[chunk_idx]
            _, samples_per_chunk, _ = stsc[min(sc_idx, len(stsc) - 1)]
            for _ in range(samples_per_chunk):
                if sample_idx >= len(stsz):
                    break
                offsets.append(offset)
                offset += stsz[sample_idx]
                sample_idx += 1
            chunk_idx += 1
        sc_idx += 1

    if len(offsets) < len(stsz):
        # Pad with last known offset progression for malformed stsc edge cases.
        last = offsets[-1] if offsets else 0
        while len(offsets) < len(stsz):
            idx = len(offsets)
            offsets.append(last)
            last += stsz[idx]

    return offsets


def _first_sample_at_or_after(pts: list[float], ts: float) -> int:
    lo, hi = 0, len(pts)
    while lo < hi:
        mid = (lo + hi) // 2
        if pts[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    return min(lo, len(pts) - 1)


def _last_sample_at_or_before(pts: list[float], ts: float) -> int:
    lo, hi = 0, len(pts)
    while lo < hi:
        mid = (lo + hi) // 2
        if pts[mid] <= ts:
            lo = mid + 1
        else:
            hi = mid
    return max(0, lo - 1)


def _keyframe_back(sync_samples: list[int], sample_idx: int) -> int:
    if not sync_samples:
        return max(0, sample_idx - 2)
    # stss stores 1-based sample numbers
    one_based = sample_idx + 1
    prev = [s for s in sync_samples if s <= one_based]
    if prev:
        return prev[-1] - 1
    return 0
