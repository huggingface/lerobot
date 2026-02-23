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
import queue
import shutil
import tempfile
import threading
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import av
import fsspec
import numpy as np
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image

# List of hardware encoders to probe for auto-selection. Availability depends on the platform and FFmpeg build.
# Determines the order of preference for auto-selection when vcodec="auto" is used.
HW_ENCODERS = [
    "h264_videotoolbox",  # macOS
    "hevc_videotoolbox",  # macOS
    "h264_nvenc",  # NVIDIA GPU
    "hevc_nvenc",  # NVIDIA GPU
    "h264_vaapi",  # Linux Intel/AMD
    "h264_qsv",  # Intel Quick Sync
]

VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1", "auto"} | set(HW_ENCODERS)


def _get_codec_options(
    vcodec: str,
    g: int | None = 2,
    crf: int | None = 30,
    preset: int | None = None,
) -> dict:
    """Build codec-specific options dict for video encoding."""
    options = {}

    # GOP size (keyframe interval) - supported by VideoToolbox and software encoders
    if g is not None and (vcodec in ("h264_videotoolbox", "hevc_videotoolbox") or vcodec not in HW_ENCODERS):
        options["g"] = str(g)

    # Quality control (codec-specific parameter names)
    if crf is not None:
        if vcodec in ("h264", "hevc", "libsvtav1"):
            options["crf"] = str(crf)
        elif vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
            quality = max(1, min(100, int(100 - crf * 2)))
            options["q:v"] = str(quality)
        elif vcodec in ("h264_nvenc", "hevc_nvenc"):
            options["rc"] = "constqp"
            options["qp"] = str(crf)
        elif vcodec in ("h264_vaapi",):
            options["qp"] = str(crf)
        elif vcodec in ("h264_qsv",):
            options["global_quality"] = str(crf)

    # Preset (only for libsvtav1)
    if vcodec == "libsvtav1":
        options["preset"] = str(preset) if preset is not None else "12"

    return options


def detect_available_hw_encoders() -> list[str]:
    """Probe PyAV/FFmpeg for available hardware video encoders."""
    available = []
    for codec_name in HW_ENCODERS:
        try:
            av.codec.Codec(codec_name, "w")
            available.append(codec_name)
        except Exception:  # nosec B110
            pass  # nosec B110
    return available


def resolve_vcodec(vcodec: str) -> str:
    """Validate vcodec and resolve 'auto' to best available HW encoder, fallback to libsvtav1."""
    if vcodec not in VALID_VIDEO_CODECS:
        raise ValueError(f"Invalid vcodec '{vcodec}'. Must be one of: {sorted(VALID_VIDEO_CODECS)}")
    if vcodec != "auto":
        logging.info(f"Using video codec: {vcodec}")
        return vcodec
    available = detect_available_hw_encoders()
    for encoder in HW_ENCODERS:
        if encoder in available:
            logging.info(f"Auto-selected video codec: {encoder}")
            return encoder
    logging.info("No hardware encoder available, falling back to software encoder 'libsvtav1'")
    return "libsvtav1"


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class VideoDecoderCache:
    """Thread-safe cache for video decoders to avoid expensive re-initialization."""

    def __init__(self):
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        """Get a cached decoder or create a new one."""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        video_path = str(video_path)

        with self._lock:
            if video_path not in self._cache:
                file_handle = fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
                self._cache[video_path] = (decoder, file_handle)

            return self._cache[video_path][0]

    def clear(self):
        """Clear the cache and close file handles."""
        with self._lock:
            for _, file_handle in self._cache.values():
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
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range
    closest_frames = (closest_frames / 255.0).type(torch.float32)

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
    preset: int | None = None,
    encoder_threads: int | None = None,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    vcodec = resolve_vcodec(vcodec)

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logging.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Get input frames
    template = "frame-" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("-")[-1].split(".")[0])
    )

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    # Define video codec options
    video_options = _get_codec_options(vcodec, g, crf, preset)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    if encoder_threads is not None:
        if vcodec == "libsvtav1":
            lp_param = f"lp={encoder_threads}"
            if "svtav1-params" in video_options:
                video_options["svtav1-params"] += f":{lp_param}"
            else:
                video_options["svtav1-params"] = lp_param
        else:
            video_options["threads"] = str(encoder_threads)

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


def concatenate_video_files(
    input_video_paths: list[Path | str], output_video_path: Path, overwrite: bool = True
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

    Note:
        - Creates a temporary directory for intermediate files that is cleaned up after use.
        - Uses ffmpeg's concat demuxer which requires all input videos to have the same
          codec, resolution, and frame rate for proper concatenation.
    """

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logging.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

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
        vcodec: str,
        pix_fmt: str,
        g: int | None,
        crf: int | None,
        preset: int | None,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        encoder_threads: int | None = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.g = g
        self.crf = crf
        self.preset = preset
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.encoder_threads = encoder_threads

    def run(self) -> None:
        from lerobot.datasets.compute_stats import RunningQuantileStats, auto_downsample_height_width

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

                # Ensure HWC uint8 numpy array
                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                        # CHW -> HWC
                        frame_data = frame_data.transpose(1, 2, 0)
                    if frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)

                # Open container on first frame (to get width/height)
                if container is None:
                    height, width = frame_data.shape[:2]
                    video_options = _get_codec_options(self.vcodec, self.g, self.crf, self.preset)
                    if self.encoder_threads is not None:
                        if self.vcodec == "libsvtav1":
                            lp_param = f"lp={self.encoder_threads}"
                            if "svtav1-params" in video_options:
                                video_options["svtav1-params"] += f":{lp_param}"
                            else:
                                video_options["svtav1-params"] = lp_param
                        else:
                            video_options["threads"] = str(self.encoder_threads)
                    Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.video_path), "w")
                    output_stream = container.add_stream(self.vcodec, self.fps, options=video_options)
                    output_stream.pix_fmt = self.pix_fmt
                    output_stream.width = width
                    output_stream.height = height
                    output_stream.time_base = Fraction(1, self.fps)

                # Encode frame with explicit timestamps
                pil_img = Image.fromarray(frame_data)
                video_frame = av.VideoFrame.from_image(pil_img)
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
            logging.error(f"Encoder thread error: {e}")
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
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        preset: int | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        self.fps = fps
        self.vcodec = resolve_vcodec(vcodec)
        self.pix_fmt = pix_fmt
        self.g = g
        self.crf = crf
        self.preset = preset
        self.queue_maxsize = queue_maxsize
        self.encoder_threads = encoder_threads

        self._frame_queues: dict[str, queue.Queue] = {}
        self._result_queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, _CameraEncoderThread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode_active = False

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        """Start encoder threads for a new episode.

        Args:
            video_keys: List of video feature keys (e.g. ["observation.images.laptop"])
            temp_dir: Base directory for temporary MP4 files
        """
        if self._episode_active:
            self.cancel_episode()

        self._dropped_frames.clear()

        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            temp_video_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = temp_video_dir / f"{video_key.replace('/', '_')}_streaming.mp4"

            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                vcodec=self.vcodec,
                pix_fmt=self.pix_fmt,
                g=self.g,
                crf=self.crf,
                preset=self.preset,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
                encoder_threads=self.encoder_threads,
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
                logging.warning(
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
                logging.warning(f"Episode finished with {count} dropped frame(s) for {video_key}.")

        # Send sentinel to all queues
        for video_key in self._frame_queues:
            self._frame_queues[video_key].put(None)

        # Wait for all threads and collect results
        for video_key in self._threads:
            self._threads[video_key].join(timeout=120)
            if self._threads[video_key].is_alive():
                logging.error(f"Encoder thread for {video_key} did not finish in time")
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
                logging.error(f"No result from encoder thread for {video_key}")
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
        if self._episode_active:
            self.cancel_episode()

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


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
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
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


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
        streaming_encoder = getattr(self.dataset, "_streaming_encoder", None)

        if streaming_encoder is not None:
            # Handle streaming encoder cleanup
            if exc_type is not None:
                streaming_encoder.cancel_episode()
            streaming_encoder.close()
        elif self.dataset.episodes_since_last_encoding > 0:
            # Handle any remaining episodes that haven't been batch encoded
            if exc_type is not None:
                logging.info("Exception occurred. Encoding remaining episodes before exit...")
            else:
                logging.info("Recording stopped. Encoding remaining episodes...")

            start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
            end_ep = self.dataset.num_episodes
            logging.info(
                f"Encoding remaining {self.dataset.episodes_since_last_encoding} episodes, "
                f"from episode {start_ep} to {end_ep - 1}"
            )
            self.dataset._batch_save_episode_video(start_ep, end_ep)

        # Finalize the dataset to properly close all writers
        self.dataset.finalize()

        # Clean up episode images if recording was interrupted (only for non-streaming mode)
        if exc_type is not None and streaming_encoder is None:
            interrupted_episode_index = self.dataset.num_episodes
            for key in self.dataset.meta.video_keys:
                img_dir = self.dataset._get_image_file_path(
                    episode_index=interrupted_episode_index, image_key=key, frame_index=0
                ).parent
                if img_dir.exists():
                    logging.debug(
                        f"Cleaning up interrupted episode images for episode {interrupted_episode_index}, camera {key}"
                    )
                    shutil.rmtree(img_dir)

        # Clean up any remaining images directory if it's empty
        img_dir = self.dataset.root / "images"
        if img_dir.exists():
            png_files = list(img_dir.rglob("*.png"))
            if len(png_files) == 0:
                shutil.rmtree(img_dir)
                logging.debug("Cleaned up empty images directory")
            else:
                logging.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False  # Don't suppress the original exception
