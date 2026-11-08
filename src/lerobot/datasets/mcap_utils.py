#!/usr/bin/env python

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

"""Helpers for the mcap dataset reader.

Everything the ``"mcap"`` storage format needs outside the reader class: topic
naming, raw tensor (de)serialization, and the on-disk path convention (one
mcap file per episode, located the same way ``videos/{key}/chunk_index`` locates
an mp4 file in the default/lance formats). The one-time, conversion-time-only
per-channel timestamp index build also lives here since it's format plumbing,
not reader logic — the reader only ever reads the index back from
``meta/episodes``, never rebuilds it (see :mod:`lerobot.datasets.mcap_backend`).
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _av_available, _mcap_available, require_package

from .utils import CHUNK_FILE_PATTERN, DATA_DIR

if TYPE_CHECKING or _mcap_available:
    from mcap.reader import make_reader
if TYPE_CHECKING or _av_available:
    import av

# One mcap file per episode, chunked the same way as the default/lance formats.
DEFAULT_MCAP_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".mcap"

# meta/episodes columns this format adds on top of dataset_from_index/dataset_to_index.
MCAP_CHUNK_INDEX_COLUMN = "mcap/chunk_index"
MCAP_FILE_INDEX_COLUMN = "mcap/file_index"
EPISODE_START_TIME_COLUMN = "episode_start_time_ns"

# Optional key -> physical mcap topic map. Datasets built by convert_lerobot_dataset_to_mcap
# don't need one (topic_for_key's convention is used directly); datasets ingested from
# externally-recorded .mcap files (arbitrary topic names, e.g. "/camera/rgb/image_raw")
# carry one so the raw files can be used as-is -- no re-encode just to rename topics.
TOPIC_MAP_PATH = "meta/mcap_topics.json"


def write_topic_map(root: Path, topics: dict[str, str]) -> None:
    path = root / TOPIC_MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topics, indent=2, sort_keys=True))


def load_topic_map(root: Path) -> dict[str, str]:
    path = root / TOPIC_MAP_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def topic_for_key(key: str) -> str:
    """Feature key -> mcap topic, e.g. ``observation.state`` -> ``/observation/state``."""
    return "/" + key.replace(".", "/")


def key_for_topic(topic: str) -> str:
    """Inverse of :func:`topic_for_key`."""
    return topic.lstrip("/").replace("/", ".")


def log_time_column(key: str) -> str:
    """The ``meta/episodes`` column holding a channel's per-episode message timestamps."""
    return f"mcap/{key}/log_time_ns"


def keyframe_index_column(key: str) -> str:
    """The ``meta/episodes`` column holding a GOP-encoded video channel's per-episode
    keyframe positions -- indices into that channel's own ``log_time_column`` array
    (mcap's unit of fetch is already a message, so there's no byte offset to store here,
    unlike the lance backend's ``kf_positions``)."""
    return f"mcap/{key}/kf_indices"


def pack_tensor(array: np.ndarray) -> bytes:
    """Pack a numpy array into raw contiguous bytes for a tabular mcap message."""
    return np.ascontiguousarray(array).tobytes()


def unpack_tensor(data: bytes, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`pack_tensor`."""
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def channel_schema_json(dtype: str, shape: tuple[int, ...]) -> bytes:
    """Self-describing schema payload for a tabular channel (external inspectability only;
    the reader never parses this back — it trusts ``meta.features[key]`` instead)."""
    return json.dumps({"dtype": dtype, "shape": list(shape)}).encode("utf-8")


def episode_mcap_path(root: Path, chunk_index: int, file_index: int) -> Path:
    return root / DEFAULT_MCAP_PATH.format(chunk_index=chunk_index, file_index=file_index)


def build_channel_time_index(mcap_path: str | Path, topics: list[str]) -> dict[str, np.ndarray]:
    """One streaming pass over ``mcap_path``; returns ``{topic: sorted log_time_ns array}``.

    No ``decoder_factories`` are used, so message payloads are never deserialized —
    only ``message.log_time`` is read. Called once by the converter at dataset
    creation time; never called by the reader (see the module docstring).
    """
    require_package("mcap", extra="mcap")
    times: dict[str, list[int]] = {topic: [] for topic in topics}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=list(topics)):
            times[channel.topic].append(message.log_time)
    return {topic: np.array(sorted(values), dtype=np.int64) for topic, values in times.items()}


# NAL unit types that carry Supplemental Enhancement Information -- metadata a
# compliant decoder is required to ignore when reconstructing picture content (ITU-T
# H.264 7.4.1 / H.265 7.4.2.1): stripping these can never change decoded pixels.
_SEI_NAL_TYPES = {"h264": frozenset({6}), "hevc": frozenset({39, 40})}  # suffix SEI included


def _annexb_nal_spans(data: bytes) -> list[tuple[int, int, int]]:
    """``(unit_start, header_start, unit_end)`` for every NAL unit in Annex-B ``data``,
    covering both 3- and 4-byte start codes. ``header_start`` is where the NAL header
    byte(s) begin (right after the start code); ``unit_start`` includes the start code
    itself, so slicing ``data[unit_start:unit_end]`` reproduces that NAL verbatim."""
    marks: list[tuple[int, int]] = []
    i = 0
    n = len(data)
    while i < n - 2:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            marks.append((i, i + 3))
            i += 3
            continue
        if i < n - 3 and data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 0 and data[i + 3] == 1:
            marks.append((i, i + 4))
            i += 4
            continue
        i += 1
    return [
        (unit_start, header_start, marks[idx + 1][0] if idx + 1 < len(marks) else n)
        for idx, (unit_start, header_start) in enumerate(marks)
    ]


def _strip_sei_nal_units(data: bytes, codec: str) -> bytes:
    """Drop SEI NAL units (H.264 type 6; HEVC types 39/40, prefix and suffix) from one
    raw Annex-B access-unit's bytes before muxing. SEI is defined by spec to carry no
    information that affects decoded picture content -- a compliant decoder produces
    identical pixels whether or not it's present -- so this is a safe transformation,
    not a lossy workaround. It matters here because a real recording was found with
    every message carrying an oversized SEI payload (confirmed: ~1MB messages, almost
    entirely SEI, versus hundreds of bytes to ~34KB for a comparable stream without
    this), which was confirmed (by direct before/after test) to disrupt synthetic
    pts/dts preservation through the MPEG-TS mux/demux round trip
    :func:`mux_and_decode_by_position` relies on for frame identity.

    Only a codec with a known SEI NAL type set is touched; anything else (or data with
    no detectable NAL structure at all) passes through unchanged. If stripping would
    leave nothing behind (a message that was genuinely SEI-only, no slice/parameter-set
    content), the original bytes are kept instead: such a message never decodes to a
    frame either way, so there's nothing to gain from stripping it, and muxing a
    zero-length packet reliably raises.
    """
    sei_types = _SEI_NAL_TYPES.get(codec)
    if not sei_types:
        return data
    spans = _annexb_nal_spans(data)
    if not spans:
        return data
    out = bytearray()
    for unit_start, header_start, unit_end in spans:
        nal_type = (data[header_start] >> 1) & 0x3F if codec == "hevc" else data[header_start] & 0x1F
        if nal_type in sei_types:
            continue
        out += data[unit_start:unit_end]
    return bytes(out) if out else data


def mux_and_decode_by_position(messages: list[bytes], codec: str) -> dict[int, av.VideoFrame]:
    """Mux raw per-access-unit packet bytes (one per position, in iteration order) into a
    throwaway in-memory MPEG-TS container -- Annex-B H.264/HEVC packetizes directly into
    MPEG-TS, with no NAL-length conversion needed, unlike mp4/mov -- tagging each with a
    synthetic pts equal to its own position, then decodes through the real container API
    (``av.open()`` + ``container.decode(stream)``). Returns ``{position: frame}`` for
    every message that produced a frame; a message that produces none (e.g. a
    parameter-set-only packet -- see ``build_gop_time_and_keyframe_index``) is simply
    absent, not an error.

    One confirmed edge case: the container's own Annex-B parser reconstructs access
    units from NAL content, not from where this function's caller happened to split
    messages -- a parameter-set-only message with no slice NAL of its own gets merged
    into whatever LATER message supplies the first slice NAL, and the merged frame is
    attributed to the EARLIER message's position (the later one is then absent, as if it
    produced nothing). The decoded picture itself is unaffected (it's genuinely that
    later message's real content); only the position/timestamp attribution shifts by one
    message. Harmless for the real-world shape this is built for (a handful of
    parameter-set messages sent once, immediately followed by the stream's first real
    picture, usually well within any reasonable ``tolerance_s``).

    This is the only decode strategy this format uses for GOP-encoded video, never a
    bare, one-packet-at-a-time ``CodecContext.decode()``, for two reasons confirmed
    empirically while building this support: (1) a bare decode assumes one message
    produces at most one frame in arrival order, which is false for B-frame content
    (decoder-side reordering); (2) independently of B-frames, a bare decode can outright
    raise ``InvalidDataError`` on packets a real container demuxes and decodes without
    issue (observed with packets from a muxer-aware encoder, as opposed to a bare
    ``CodecContext`` encoder) -- i.e. it isn't even a reliable way to inspect arbitrary
    GOP packets, only a real container decode is.

    The synthetic pts/dts survive the mux/demux round trip exactly -- confirmed
    empirically: ``round(marker * frame.time_base)`` always recovers the integer
    originally assigned, regardless of the container format's own internal clock (mpegts
    always rescales to a 90kHz clock, for instance) -- which is what lets each decoded
    frame be matched back to the exact position it came from, independent of what order
    the container/decoder actually emits frames in. That sidesteps needing to
    reconstruct true presentation order at all: every position is looked up by identity,
    not by position in the output sequence.

    Neither ``frame.pts`` nor ``frame.dts`` is universally reliable on its own, though --
    confirmed on independent real failure modes while building this: a handful of frames
    flushed at the very end of a stream (no new packet feeds them, so they have no
    decode-time to report) reliably lose ``dts`` while keeping ``pts``; separately, before
    the SEI-stripping below was added, most non-keyframe frames of two (of four) camera
    streams in a real multi-camera recording lost ``pts`` specifically while (confirmed)
    keeping ``dts``. Either field recovers the same integer when both are intact (they're
    rescaled through the same container clock), so this uses whichever one survived, and
    raises -- rather than silently mis-associating content -- only in the degenerate case
    where a single frame loses both and the exact message it belongs to truly cannot be
    recovered (confirmed to still happen for that same recording's episode-opening
    keyframe specifically, even after stripping SEI -- large enough on its own, from
    genuine picture data, that whatever about the *first* muxed packet triggers this
    wasn't narrowed down further; the pts/dts fallback above, and this function's raise
    when both are gone, are what keep that case from silently producing wrong content).

    Every message is passed through :func:`_strip_sei_nal_units` first: real encoders can
    attach an SEI payload many times the size of the actual compressed picture (see that
    function's docstring for the confirmed real-world numbers), which is what triggers
    the pts-loss failure mode above for the streams that hit it. SEI carries no
    information that affects decoded picture content by spec, so removing it can't
    change what a frame decodes to -- only (as observed) whether its container-level
    identity metadata survives the round trip.
    Used by both :func:`build_gop_time_and_keyframe_index` (ingest time) and
    :meth:`lerobot.datasets.mcap_backend.McapDatasetReader._decode_gop_window_muxed`
    (read time), with different message sources.
    """
    require_package("av", extra="av-dep")
    buf = io.BytesIO()
    out_container = av.open(buf, mode="w", format="mpegts")
    out_stream = out_container.add_stream(codec)
    for position, data in enumerate(messages):
        packet = av.Packet(_strip_sei_nal_units(data, codec))
        packet.pts = position
        packet.dts = position
        packet.time_base = Fraction(1, 1)
        packet.stream = out_stream
        out_container.mux(packet)
    out_container.close()

    buf.seek(0)
    with av.open(buf) as in_container:
        in_stream = in_container.streams.video[0]
        return _recover_frames_by_position(in_container.decode(in_stream), codec, len(messages))


def _recover_frames_by_position(
    frames: Iterable[av.VideoFrame], codec: str, n_messages: int
) -> dict[int, av.VideoFrame]:
    """The identity-recovery half of :func:`mux_and_decode_by_position`, split out
    purely so it's directly testable against a scripted frame sequence without needing
    a real mux/demux round trip (see ``tests/datasets/test_mcap_backend.py``)."""
    frames_by_position: dict[int, av.VideoFrame] = {}
    for frame in frames:
        marker = frame.pts if frame.pts is not None else frame.dts
        if marker is None:
            raise ValueError(
                f"A decoded frame (codec={codec!r}, {n_messages} input messages) lost "
                "both its pts and dts through the mux/demux round trip -- cannot recover "
                "which original message it came from."
            )
        position = round(marker * frame.time_base)
        if position in frames_by_position:
            raise ValueError(
                f"Two decoded frames (codec={codec!r}, {n_messages} input messages) both "
                f"recovered position {position} -- the pts/dts identity mapping is not unique "
                "for this stream, refusing to silently pick one."
            )
        frames_by_position[position] = frame
    return frames_by_position


def build_gop_time_and_keyframe_index(
    mcap_path: str | Path, topic: str, codec: str
) -> tuple[np.ndarray, np.ndarray, bool, tuple[int, int, int]]:
    """``(log_time_ns, kf_positions, has_bframes, frame_shape)`` for one continuous
    GOP-encoded video channel, built together from a single decode pass over its
    messages. ``kf_positions`` indexes into the returned ``log_time_ns`` array -- the
    mcap-native analogue of
    :func:`lerobot.datasets.lance_utils.build_video_byte_index`'s byte-offset
    ``kf_indices``, except positions are over messages rather than file bytes.
    ``has_bframes`` records whether any decoded frame's picture type is B, which the
    reader uses to pick its decode strategy (see ``McapDatasetReader._gop_has_bframes``).
    ``frame_shape`` is ``(channels, height, width)`` from the first successfully decoded
    frame, so a separate shape-peeking decode pass over the same messages isn't needed.

    Unlike :func:`build_channel_time_index` (which records every message's timestamp
    without decoding), this one decodes every message (via :func:`mux_and_decode_by_position`
    -- see its docstring for the full mechanism and why) and keeps only the timestamps of
    messages that actually produce a frame. Real encoders commonly front-load
    parameter-set-only packets (VPS/SPS/PPS for HEVC, SPS/PPS for H.264) before any
    picture data, which produce no frame to serve a grid match, so they're dropped from
    the index entirely rather than counted against the "must start with a keyframe"
    check below. Called once by the converter; never by the reader.
    """
    require_package("mcap", extra="mcap")
    require_package("av", extra="av-dep")

    message_log_times: list[int] = []
    raw_messages: list[bytes] = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(topics=[topic], log_time_order=True):
            raw_messages.append(message.data)
            message_log_times.append(message.log_time)

    frames_by_position = mux_and_decode_by_position(raw_messages, codec)
    has_bframes = any(
        frame.pict_type == av.video.frame.PictureType.B for frame in frames_by_position.values()
    )
    frame_shape: tuple[int, int, int] | None = None
    if frames_by_position:
        first_frame = next(iter(frames_by_position.values()))
        frame_shape = (3, first_frame.height, first_frame.width)

    times: list[int] = []
    kf_positions: list[int] = []
    n = len(message_log_times)
    i = 0
    while i < n:
        frame = frames_by_position.get(i)
        if frame is None:
            i += 1
            continue  # zero-output message (e.g. parameter-set-only), dropped entirely
        # Any positions immediately following this one that are themselves absent were
        # merged into THIS decoded frame by the container's own Annex-B parser (see
        # mux_and_decode_by_position's docstring) -- the raw message that actually
        # supplied the completing slice NAL is the LAST of that run, not this
        # (possibly parameter-set-only) position itself, so ITS timestamp is what must
        # be recorded: using this position's own would desync every later grid index
        # against this channel's real message times by one slot (confirmed empirically).
        j = i + 1
        while j < n and j not in frames_by_position:
            j += 1
        if frame.key_frame:
            kf_positions.append(len(times))
        elif not kf_positions:
            raise ValueError(
                f"Video channel on topic '{topic}' in {mcap_path} must start with a keyframe "
                "(open GOPs are not supported)."
            )
        times.append(message_log_times[j - 1])
        i = j
    if not kf_positions or frame_shape is None:
        raise ValueError(f"No keyframe found for video channel on topic '{topic}' in {mcap_path}.")
    return np.array(times, dtype=np.int64), np.array(kf_positions, dtype=np.int64), has_bframes, frame_shape


def localize_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> Path:
    """v1 is local-disk only: raise immediately so ``storage.py``'s format-probe loop
    (:func:`lerobot.datasets.storage.localize_remote_root`) skips this format and tries
    others, instead of implementing object-store support here.
    """
    raise FileNotFoundError("storage_format='mcap' does not support object-store roots yet.")
