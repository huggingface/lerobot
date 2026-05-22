#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import contextlib
import glob
import importlib
import logging
import os
import queue
import shutil
import tempfile
import threading
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import av
import fsspec
import numpy as np
import pyarrow as pa
import torch
from datasets.features.features import register_feature
from PIL import Image

from lerobot.configs import (
    DepthEncoderConfig,
    VideoEncoderConfig,
    camera_encoder_defaults,
    depth_encoder_defaults,
)
from lerobot.utils.import_utils import get_safe_default_video_backend

from .depth_utils import quantize_depth
from .pyav_utils import get_pix_fmt_channels

logger = logging.getLogger(__name__)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available
            in the platform; otherwise, defaults to "pyav". The legacy value "video_reader" is
            accepted for one release as an alias for "pyav" and will be removed in a future version.
        return_uint8 (bool): If True, return raw uint8 frames without float32 normalization.
            This reduces memory for DataLoader IPC; normalization can be done on GPU afterward.

    Returns:
        torch.Tensor: Decoded frames (float32 in [0,1] by default, or uint8 if return_uint8=True).

    Currently supports torchcodec on cpu and pyav.
    """
    if backend != "pyav" and is_depth:
        logger.warning("Decoding depth maps is only supported with the 'pyav' backend.")
        # We do not actually return uint8 here, but we avoid the 255 normalization step.
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s, return_uint8=True, is_depth=True)

    if backend is None:
        backend = get_safe_default_video_backend()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    elif backend == "pyav":
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    elif backend == "video_reader":
        logger.warning("backend='video_reader' is deprecated and now aliases to 'pyav'.")
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video using PyAV.

    This is the fallback decoder for platforms where torchcodec has no wheel (currently macOS
    x86_64 and linux armv7l — see the torchcodec block in pyproject.toml for the full matrix).
    On supported platforms, prefer `decode_video_frames_torchcodec`, which is faster and supports
    accurate seek.

    PyAV doesn't support accurate seek: we seek to the nearest preceding keyframe and decode
    forward until we have covered the requested timestamp range. The number of key frames in a
    video can be adjusted at encoding time to trade off decoding speed against file size.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps (in seconds) to extract frames for.
        tolerance_s: Allowed deviation in seconds between a queried timestamp and the closest
            decoded frame.
        log_loaded_timestamps: When True, log every decoded frame's timestamp at INFO level.
        return_uint8: When True, return raw uint8 frames (C, H, W). Otherwise, return float32 in
            [0, 1] range.

    Returns:
        torch.Tensor of shape (len(timestamps), C, H, W).
    """
    # TODO(rcadene): also load audio stream at the same time
    video_path = str(video_path)

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    # Seek + decode. `container.seek(offset)` with no `stream` argument expects the offset in
    # av.time_base units (microseconds). `backward=True` lands us on the nearest keyframe at or
    # before `first_ts`, so we can then decode forward until we cover `last_ts`. See:
    # https://pyav.basswood-io.com/docs/stable/api/container.html#av.container.InputContainer.seek
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(int(first_ts * av.time_base), backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logger.info(f"frame loaded at timestamp={current_ts:.4f}")
            if is_depth:
                arr = frame.to_ndarray(format="gray12le")  # (H, W) uint12
                loaded_frames.append(torch.from_numpy(arr).unsqueeze(0).contiguous())
            else:
                arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                # Convert to CHW uint8 to match torchcodec's output layout.
                loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise FrameTimestampError(
            f"No frames could be decoded from {video_path} in the timestamp range [{first_ts}, {last_ts}]."
        )

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_t}"
            f"\nvideo: {video_path}"
            f"\nbackend: pyav"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts_t[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )

    if return_uint8:
        return closest_frames

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames


DEFAULT_DECODER_CACHE_SIZE = 100
"""Default LRU capacity for :class:`VideoDecoderCache`.

Sized to comfortably hold a small rolling window of episodes worth of decoders
(typical recipes: 2-4 cameras per episode × tens of episodes in flight) while
bounding host RAM. Each cached entry retains a torchcodec ``VideoDecoder`` plus
an open ``fsspec`` file handle — on the order of a few MB per entry. Override
via the ``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` env var or by passing ``max_size``
to the constructor (``None`` restores the legacy unbounded behaviour).
"""


def _default_max_cache_size() -> int | None:
    raw = os.environ.get("LEROBOT_VIDEO_DECODER_CACHE_SIZE")
    if raw is None:
        return DEFAULT_DECODER_CACHE_SIZE
    raw = raw.strip().lower()
    if raw in ("", "none", "unbounded", "-1"):
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"LEROBOT_VIDEO_DECODER_CACHE_SIZE must be an integer, 'none', or '-1'; got {raw!r}"
        ) from e
    if value <= 0:
        raise ValueError(f"LEROBOT_VIDEO_DECODER_CACHE_SIZE must be positive; got {value}")
    return value


class VideoDecoderCache:
    """Thread-safe LRU cache for torchcodec ``VideoDecoder`` instances.

    Cached entries hold a ``VideoDecoder`` plus the open ``fsspec`` file handle
    backing it. When the cache is full and a new path is requested, the
    least-recently-used entry is evicted and its file handle is closed. This
    bounds host-RAM growth when iterating over datasets with many distinct
    video files (otherwise each ``DataLoader`` worker pins every decoder it has
    ever opened until the process exits).

    Args:
        max_size: Maximum number of decoders to retain. ``None`` disables
            eviction and restores legacy unbounded behaviour. Defaults to the
            value of ``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` if set, otherwise
            :data:`DEFAULT_DECODER_CACHE_SIZE`.
    """

    _SENTINEL: ClassVar[object] = object()

    def __init__(self, max_size: int | None | object = _SENTINEL):
        if max_size is VideoDecoderCache._SENTINEL:
            max_size = _default_max_cache_size()
        if max_size is not None and max_size <= 0:
            raise ValueError(f"max_size must be positive or None; got {max_size}")
        self.max_size: int | None = max_size  # type: ignore[assignment]
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()

    def __contains__(self, video_path: object) -> bool:
        with self._lock:
            return str(video_path) in self._cache

    def get_decoder(self, video_path: str):
        """Get a cached decoder or create a new one, evicting LRU if at capacity."""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError(
                "'torchcodec' is required but not installed. "
                "Install it with: pip install 'lerobot[dataset]' (or uv pip install 'lerobot[dataset]')"
            )

        video_path = str(video_path)

        with self._lock:
            entry = self._cache.get(video_path)
            if entry is not None:
                self._cache.move_to_end(video_path)
                return entry[0]

            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
            except Exception:
                file_handle.close()
                raise
            self._cache[video_path] = (decoder, file_handle)

            # Evict LRU entries until we are back under the cap. We close
            # evicted file handles immediately; the associated ``VideoDecoder``
            # is released to the GC when its last reference goes away.
            if self.max_size is not None:
                while len(self._cache) > self.max_size:
                    _evicted_path, (_evicted_decoder, evicted_handle) = self._cache.popitem(last=False)
                    with contextlib.suppress(Exception):
                        evicted_handle.close()

            return decoder

    def clear(self):
        """Clear the cache and close all file handles."""
        with self._lock:
            for _, file_handle in self._cache.values():
                with contextlib.suppress(Exception):
                    file_handle.close()
            self._cache.clear()

    def size(self) -> int:
        """Return the number of cached decoders."""
        with self._lock:
            return len(self._cache)


class FrameTimestampError(ValueError):
    """Helper error to indicate the retrieved timestamps exceed the queried ones"""

    pass


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
    return_uint8: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps to extract frames.
        tolerance_s: Allowed deviation in seconds for frame retrieval.
        log_loaded_timestamps: Whether to log loaded timestamps.
        decoder_cache: Optional decoder cache instance. Uses default if None.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    if decoder_cache is None:
        decoder_cache = _default_decoder_cache

    # Use cached decoder instead of creating new one each time
    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []

    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logger.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    if return_uint8:
        return closest_frames

    # convert to float32 in [0,1] range
    closest_frames = (closest_frames / 255.0).type(torch.float32)
    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    video_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    *,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    if video_encoder is None:
        video_encoder = camera_encoder_defaults()
    vcodec = video_encoder.vcodec
    pix_fmt = video_encoder.pix_fmt

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Get input frames
    suffix = ".png" if not isinstance(video_encoder, DepthEncoderConfig) else ".tiff"
    template = "frame-" + ("[0-9]" * 6) + suffix
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("-")[-1].split(".")[0])
    )

    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    video_options = video_encoder.get_codec_options(encoder_threads, as_strings=True)

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python's logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            with Image.open(input_data) as input_image:
                input_image = input_image.convert("RGB")
                input_frame = av.VideoFrame.from_image(input_image)
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def reencode_video(
    input_video_path: Path | str,
    output_video_path: Path | str,
    camera_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> None:
    """Re-encode a video file, optionally trimming it to ``[start_time_s, end_time_s)``.

    Args:
        input_video_path: Existing video file to read.
        output_video_path: Path for the re-encoded file.
        camera_encoder: Encoder configuration. Defaults to :func:`camera_encoder_defaults`.
        encoder_threads: Optional thread count forwarded to :meth:`VideoEncoderConfig.get_codec_options`.
        log_level: libav log level while encoding, or ``None`` to leave logging unchanged. Defaults to WARNING.
        overwrite: When ``False`` and ``output_video_path`` already exists, skip and log a warning.
        start_time_s: When set, trim the output to start at this timestamp (seconds).
        end_time_s: When set, trim the output to end at this timestamp (seconds, exclusive).
    """

    camera_encoder = camera_encoder or camera_encoder_defaults()

    if (start_time_s is not None and start_time_s < 0) or (end_time_s is not None and end_time_s < 0):
        raise ValueError(f"Trim times must be non-negative, got start={start_time_s}, end={end_time_s}.")
    if start_time_s is not None and end_time_s is not None and end_time_s <= start_time_s:
        raise ValueError(f"end_time_s ({end_time_s}) must be greater than start_time_s ({start_time_s}).")

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping re-encode.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    video_options = camera_encoder.get_codec_options(encoder_threads, as_strings=True)
    vcodec = camera_encoder.vcodec
    pix_fmt = camera_encoder.pix_fmt

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    try:
        with av.open(input_video_path, mode="r") as src:
            try:
                in_stream = src.streams.video[0]
            except IndexError as e:
                raise ValueError(f"No video stream in {input_video_path}") from e

            fps = (
                in_stream.base_rate
            )  # We allow fractional fps though LeRobotDataset only supports integer fps
            width = int(in_stream.width)
            height = int(in_stream.height)

            # Seek to the keyframe at or before start_time_s to avoid reading from the start.
            if start_time_s is not None:
                src.seek(int(start_time_s * av.time_base), backward=True)

            with av.open(
                tmp_output_video_path,
                mode="w",
                options={
                    "movflags": "faststart"
                },  # faststart is to move the metadata to the beginning of the file to speed up loading
            ) as dst:
                out_stream = dst.add_stream(vcodec, fps, options=video_options)
                out_stream.pix_fmt = pix_fmt
                out_stream.width = width
                out_stream.height = height

                for frame in src.decode(in_stream):
                    frame_time_s = frame.time
                    if start_time_s is not None and frame_time_s < start_time_s:
                        continue
                    if end_time_s is not None and frame_time_s >= end_time_s:
                        break
                    frame = frame.reformat(width=width, height=height, format=pix_fmt)
                    if start_time_s is not None:
                        frame.pts = None  # reset timestamps so the trimmed output starts at t=0
                    packet = out_stream.encode(frame)
                    if packet:
                        dst.mux(packet)

                packet = out_stream.encode()
                if packet:
                    dst.mux(packet)

        shutil.move(tmp_output_video_path, output_video_path)
    except Exception:
        Path(tmp_output_video_path).unlink(missing_ok=True)
        raise
    finally:
        if log_level is not None:
            av.logging.restore_default_callback()

    if not output_video_path.exists():
        raise OSError(f"Video re-encoding did not work. File not found: {output_video_path}.")


def concatenate_video_files(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    overwrite: bool = True,
    compatibility_check: bool = False,
):
    """
    Concatenate multiple video files into a single video file using pyav.

    This function takes a list of video input file paths and concatenates them into a single
    output video file. It uses ffmpeg's concat demuxer with stream copy mode for fast
    concatenation without re-encoding.

    Args:
        input_video_paths: Ordered list of input video file paths to concatenate.
        output_video_path: Path to the output video file.
        overwrite: Whether to overwrite the output video file if it already exists. Default is True.
        compatibility_check: Whether to check if the input videos are compatible. Default is False.

    Note:
        - Creates a temporary directory for intermediate files that is cleaned up after use.
        - Uses ffmpeg's concat demuxer which requires all input videos to have the same
          codec, resolution, and frame rate for proper concatenation.
    """

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    # This check may be skipped at recording time as videos are encoded with the same encoder config.
    if compatibility_check:
        reference_video_info = get_video_info(input_video_paths[0])
        for input_path in input_video_paths[1:]:
            video_info = get_video_info(input_path)
            if (
                video_info["video.height"] != reference_video_info["video.height"]
                or video_info["video.width"] != reference_video_info["video.width"]
                or video_info["video.fps"] != reference_video_info["video.fps"]
                or video_info["video.codec"] != reference_video_info["video.codec"]
                or video_info["video.pix_fmt"] != reference_video_info["video.pix_fmt"]
            ):
                raise ValueError(
                    f"Input video {input_path} is not compatible with the reference video {input_video_paths[0]}."
                )

    # Create a temporary .ffconcat file to list the input video paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as tmp_concatenate_file:
        tmp_concatenate_file.write("ffconcat version 1.0\n")
        for input_path in input_video_paths:
            tmp_concatenate_file.write(f"file '{str(input_path.resolve())}'\n")
        tmp_concatenate_file.flush()
        tmp_concatenate_path = tmp_concatenate_file.name

    # Create input and output containers
    input_container = av.open(
        tmp_concatenate_path, mode="r", format="concat", options={"safe": "0"}
    )  # safe = 0 allows absolute paths as well as relative paths

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    output_container = av.open(
        tmp_output_video_path, mode="w", options={"movflags": "faststart"}
    )  # faststart is to move the metadata to the beginning of the file to speed up loading

    # Replicate input streams in output container
    stream_map = {}
    for input_stream in input_container.streams:
        if input_stream.type in ("video", "audio", "subtitle"):  # only copy compatible streams
            stream_map[input_stream.index] = output_container.add_stream_from_template(
                template=input_stream, opaque=True
            )

            # set the time base to the input stream time base (missing in the codec context)
            stream_map[input_stream.index].time_base = input_stream.time_base

    # Demux + remux packets (no re-encode)
    for packet in input_container.demux():
        # Skip packets from un-mapped streams
        if packet.stream.index not in stream_map:
            continue

        # Skip demux flushing packets
        if packet.dts is None:
            continue

        output_stream = stream_map[packet.stream.index]
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    shutil.move(tmp_output_video_path, output_video_path)
    Path(tmp_concatenate_path).unlink()


class _CameraEncoderThread(threading.Thread):
    """A thread that encodes video frames streamed via a queue into an MP4 file.

    One instance is created per camera per episode. Frames are received as numpy arrays
    from the main thread, encoded in real-time using PyAV (which releases the GIL during
    encoding), and written to disk. Stats are computed incrementally using
    RunningQuantileStats and returned via result_queue.
    """

    def __init__(
        self,
        video_path: Path,
        fps: int,
        video_encoder: VideoEncoderConfig,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        encoder_threads: int | None = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.fps = fps
        self.video_encoder = video_encoder
        self.is_depth = isinstance(video_encoder, DepthEncoderConfig)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.encoder_threads = encoder_threads

    def run(self) -> None:
        from .compute_stats import RunningQuantileStats, auto_downsample_height_width

        container = None
        output_stream = None
        stats_tracker = RunningQuantileStats()
        frame_count = 0

        try:
            logging.getLogger("libav").setLevel(av.logging.WARNING)

            while True:
                try:
                    frame_data = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if frame_data is None:
                    # Sentinel: flush and close
                    break

                # Ensure HWC (RGB or depth) uint8 (RGB only) numpy array
                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] in (1, 3):
                        # CHW -> HWC
                        frame_data = frame_data.transpose(1, 2, 0)
                    if not self.is_depth and frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)

                # Open container on first frame (to get width/height)
                if container is None:
                    height, width = frame_data.shape[:2]
                    Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.video_path), "w")
                    output_stream = container.add_stream(
                        self.video_encoder.vcodec,
                        self.fps,
                        options=self.video_encoder.get_codec_options(self.encoder_threads, as_strings=True),
                    )
                    output_stream.pix_fmt = self.video_encoder.pix_fmt
                    output_stream.width = width
                    output_stream.height = height
                    output_stream.time_base = Fraction(1, self.fps)

                # Encode frame with explicit timestamps
                if not self.is_depth:
                    pil_img = Image.fromarray(frame_data)
                    video_frame = av.VideoFrame.from_image(pil_img)
                else:
                    video_frame = quantize_depth(
                        frame_data,
                        depth_min=self.video_encoder.depth_min,
                        depth_max=self.video_encoder.depth_max,
                        shift=self.video_encoder.shift,
                        use_log=self.video_encoder.use_log,
                        video_backend=self.video_encoder.video_backend,
                    )
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, self.fps)
                packet = output_stream.encode(video_frame)
                if packet:
                    container.mux(packet)

                # Update stats with downsampled frame (per-channel stats like compute_episode_stats)
                img_chw = frame_data.transpose(2, 0, 1)  # HWC -> CHW
                img_downsampled = auto_downsample_height_width(img_chw)
                # Reshape CHW to (H*W, C) for per-channel stats
                channels = img_downsampled.shape[0]
                img_for_stats = img_downsampled.transpose(1, 2, 0).reshape(-1, channels)
                stats_tracker.update(img_for_stats)

                frame_count += 1

            # Flush encoder
            if output_stream is not None:
                packet = output_stream.encode()
                if packet:
                    container.mux(packet)

            if container is not None:
                container.close()

            av.logging.restore_default_callback()

            # Get stats and put on result queue
            if frame_count >= 2:
                stats = stats_tracker.get_statistics()
                self.result_queue.put(("ok", stats))
            else:
                self.result_queue.put(("ok", None))

        except Exception as e:
            logger.error(f"Encoder thread error: {e}")
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            self.result_queue.put(("error", str(e)))


class StreamingVideoEncoder:
    """Manages per-camera encoder threads for real-time video encoding during recording.

    Instead of writing frames as PNG images and then encoding to MP4 at episode end,
    this class streams frames directly to encoder threads, eliminating the
    PNG round-trip and making save_episode() near-instant.

    Uses threading instead of multiprocessing to avoid the overhead of pickling large
    numpy arrays through multiprocessing.Queue. PyAV's encode() releases the GIL,
    so encoding runs in parallel with the main recording loop.
    """

    def __init__(
        self,
        fps: int,
        camera_encoder: VideoEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        """
        Args:
            fps: Frames per second for the output videos.
            camera_encoder: Video encoder settings applied to all cameras.
                When ``None``, :func:`camera_encoder_defaults` is used.
            encoder_threads: Number of encoder threads (global setting).
                ``None`` lets the codec decide.
            queue_maxsize: Max frames to buffer per camera before
                back-pressure drops frames.
        """
        self.fps = fps
        self._camera_encoder = camera_encoder or camera_encoder_defaults()
        self._depth_encoder = depth_encoder or depth_encoder_defaults()
        self._encoder_threads = encoder_threads
        self.queue_maxsize = queue_maxsize

        self._frame_queues: dict[str, queue.Queue] = {}
        self._result_queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, _CameraEncoderThread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(
        self, video_keys: list[str], temp_dir: Path, depth_video_keys: list[str] | None = None
    ) -> None:
        """Start encoder threads for a new episode.

        Args:
            video_keys: List of video feature keys (e.g. ["observation.images.laptop"])
            temp_dir: Base directory for temporary MP4 files
            depth_video_keys: List of video feature keys that carry depth maps (e.g.
                ["observation.images.laptop_depth"]).  Defaults to ``[]`` (no depth keys).
        """
        if self._episode_active:
            self.cancel_episode()

        self._dropped_frames.clear()

        if depth_video_keys is None:
            depth_video_keys = []

        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            temp_video_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = temp_video_dir / f"{video_key.replace('/', '_')}_streaming.mp4"

            encoder = self._depth_encoder if video_key in depth_video_keys else self._camera_encoder
            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                video_encoder=encoder,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
                encoder_threads=self._encoder_threads,
            )
            encoder_thread.start()

            self._frame_queues[video_key] = frame_queue
            self._result_queues[video_key] = result_queue
            self._threads[video_key] = encoder_thread
            self._stop_events[video_key] = stop_event
            self._video_paths[video_key] = video_path

        self._episode_active = True

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """Feed a frame to the encoder for a specific camera.

        A copy of the image is made before enqueueing to prevent race conditions
        with camera drivers that may reuse buffers. If the encoder queue is full
        (encoder can't keep up), the frame is dropped with a warning instead of
        crashing the recording session.

        Args:
            video_key: The video feature key
            image: numpy array in (H,W,C) or (C,H,W) format, uint8 or float

        Raises:
            RuntimeError: If the encoder thread has crashed
        """
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        thread = self._threads[video_key]
        if not thread.is_alive():
            # Check for error
            try:
                status, msg = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")

        try:
            self._frame_queues[video_key].put(image.copy(), timeout=0.1)
        except queue.Full:
            self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
            count = self._dropped_frames[video_key]
            # Log periodically to avoid spam (1st, then every 10th)
            if count == 1 or count % 10 == 0:
                logger.warning(
                    f"Encoder queue full for {video_key}, dropped {count} frame(s). "
                    f"Consider using vcodec='auto' for hardware encoding or increasing encoder_queue_maxsize."
                )

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """Finish encoding the current episode.

        Sends sentinel values, waits for encoder threads to complete,
        and collects results.

        Returns:
            Dict mapping video_key to (mp4_path, stats_dict_or_None)
        """
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")

        results = {}

        # Report dropped frames
        for video_key, count in self._dropped_frames.items():
            if count > 0:
                logger.warning(f"Episode finished with {count} dropped frame(s) for {video_key}.")

        # Send sentinel to all queues
        for video_key in self._frame_queues:
            self._frame_queues[video_key].put(None)

        # Wait for all threads and collect results
        for video_key in self._threads:
            self._threads[video_key].join(timeout=120)
            if self._threads[video_key].is_alive():
                logger.error(f"Encoder thread for {video_key} did not finish in time")
                self._stop_events[video_key].set()
                self._threads[video_key].join(timeout=5)
                results[video_key] = (self._video_paths[video_key], None)
                continue

            try:
                status, data = self._result_queues[video_key].get(timeout=5)
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} failed: {data}")
                results[video_key] = (self._video_paths[video_key], data)
            except queue.Empty:
                logger.error(f"No result from encoder thread for {video_key}")
                results[video_key] = (self._video_paths[video_key], None)

        self._cleanup()
        self._episode_active = False
        return results

    def cancel_episode(self) -> None:
        """Cancel the current episode, stopping encoder threads and cleaning up."""
        if not self._episode_active:
            return

        # Signal all threads to stop
        for video_key in self._stop_events:
            self._stop_events[video_key].set()

        # Wait for threads to finish
        for video_key in self._threads:
            self._threads[video_key].join(timeout=5)

            # Clean up temp MP4 files
            video_path = self._video_paths.get(video_key)
            if video_path is not None and video_path.exists():
                shutil.rmtree(str(video_path.parent), ignore_errors=True)

        self._cleanup()
        self._episode_active = False

    def close(self) -> None:
        """Close the encoder, canceling any in-progress episode."""
        if self._closed:
            return
        if self._episode_active:
            self.cancel_episode()
        self._closed = True

    def _cleanup(self) -> None:
        """Clean up queues and thread tracking dicts."""
        for q in self._frame_queues.values():
            with contextlib.suppress(Exception):
                while not q.empty():
                    q.get_nowait()
        self._frame_queues.clear()
        self._result_queues.clear()
        self._threads.clear()
        self._stop_events.clear()
        self._video_paths.clear()


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(
    video_path: Path | str,
    video_encoder: VideoEncoderConfig | None = None,
) -> dict:
    """Build the ``video.*`` / ``audio.*`` info dict persisted in ``info.json``.

    Args:
        video_path: Path to the encoded video file to probe.
        video_encoder: If provided, record the exact encoder settings used to encode this
            video. Stream-derived values take precedence — encoder fields are only written for keys
            not already populated from the video file itself.
    """
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)
        video_info["video.channels"] = get_pix_fmt_channels(video_stream.pix_fmt)

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    # Add additional encoder configuration if provided
    if video_encoder is not None:
        for field_name, field_value in asdict(video_encoder).items():
            # vcodec is already populated from the video stream
            if field_name == "vcodec":
                continue
            video_info.setdefault(f"video.{field_name}", field_value)

    video_info["is_depth_map"] = isinstance(video_encoder, DepthEncoderConfig)

    return video_info


def get_video_duration_in_s(video_path: Path | str) -> float:
    """
    Get the duration of a video file in seconds using PyAV.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration of the video in seconds.
    """
    with av.open(str(video_path)) as container:
        # Get the first video stream
        video_stream = container.streams.video[0]
        # Calculate duration: stream.duration * stream.time_base gives duration in seconds
        if video_stream.duration is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            # Fallback to container duration if stream duration is not available
            duration = float(container.duration / av.time_base)
    return duration


class VideoEncodingManager:
    """
    Context manager that ensures proper video encoding and data cleanup even if exceptions occur.

    This manager handles:
    - Batch encoding for any remaining episodes when recording interrupted
    - Cleaning up temporary image files from interrupted episodes
    - Removing empty image directories

    Args:
        dataset: The LeRobotDataset instance
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        writer = self.dataset.writer
        if writer is not None:
            if exc_type is not None and writer._streaming_encoder is not None:
                writer.cancel_pending_videos()

            # finalize() handles flush_pending_videos + parquet + metadata
            self.dataset.finalize()

            # Clean up episode images if recording was interrupted (only for non-streaming mode)
            if exc_type is not None and writer._streaming_encoder is None:
                writer.cleanup_interrupted_episode(self.dataset.num_episodes)
        else:
            self.dataset.finalize()

        # Clean up any remaining images directory if it's empty
        img_dir = self.dataset.root / "images"
        if img_dir.exists():
            png_files = list(img_dir.rglob("*.png"))
            if len(png_files) == 0:
                shutil.rmtree(img_dir)
                logger.debug("Cleaned up empty images directory")
            else:
                logger.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False  # Don't suppress the original exception
