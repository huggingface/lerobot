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
"""Keyframe extraction for the annotation pipeline.

Modules attach decoded camera frames to their VLM prompts so the model can
ground subtask decomposition, interjection scenarios, and VQA in actual
visual content. The pipeline shares one provider across modules and one
episode at a time, with a small per-episode cache so multiple modules
querying the same timestamp pay decode cost once.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import PIL.Image
import torch

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.import_utils import get_safe_default_codec

from .reader import EpisodeRecord

logger = logging.getLogger(__name__)


class FrameProvider(Protocol):
    """Decodes camera frames at episode-relative timestamps."""

    @property
    def camera_keys(self) -> list[str]:
        """All ``observation.images.*`` feature keys this provider can decode."""

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
    ) -> list[Any]:
        """Return one decoded frame per timestamp from ``camera_key`` (or default).

        Frames are ``torch.Tensor`` (``C, H, W`` uint8) — the shape
        :func:`lerobot.datasets.video_utils.decode_video_frames` returns.
        :func:`to_image_blocks` converts them to PIL only at the VLM-message
        boundary.

        Empty list if the camera is unavailable. ``camera_key=None`` falls back
        to the provider's default camera so existing single-camera callers
        (the ``plan`` and ``interjections`` modules) keep working unchanged.
        """

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        """Return up to ``max_frames`` decoded frames covering the whole episode.

        Sampling is uniform across the episode duration. Frames are
        ``torch.Tensor`` (``C, H, W`` uint8); :func:`to_video_block` wraps
        them into one ``{"type":"video", "video":<list>}`` block for a
        Qwen-VL-compatible model that pools temporally itself. Empty list if
        no camera available.
        """


@dataclass
class _NullProvider:
    """No-op provider used when the dataset has no video keys or in tests."""

    @property
    def camera_keys(self) -> list[str]:
        return []

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
    ) -> list[Any]:
        return []

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        return []


def null_provider() -> FrameProvider:
    return _NullProvider()


@dataclass
class VideoFrameProvider:
    """Decodes frames from the dataset's ``observation.images.*`` streams.

    By default the *first* camera key is used for the ``plan`` module
    (subtask decomposition) and the ``interjections`` module (interjection
    scenarios) — those prompts care about *what is happening*, not which
    angle. The ``vqa`` module instead iterates over every camera in
    :attr:`camera_keys` so each frame's
    grounded answer (bbox/keypoint/...) is tagged with the camera it was
    grounded against.

    ``camera_key`` overrides the default-camera choice but does not restrict
    :attr:`camera_keys`. Pass ``camera_key`` explicitly to ``frames_at`` /
    ``video_for_episode`` to read a non-default stream.

    Caches up to ``cache_size`` decoded frames per process to keep
    co-timestamped ``interjections`` + ``plan`` plan-update calls cheap.
    """

    root: Path
    camera_key: str | None = None
    tolerance_s: float = 1e-2
    cache_size: int = 256
    # Keyframe decode backend. When ``None``, decoding tries the platform
    # default (torchcodec when installed) then falls back to the ffmpeg CLI.
    # Set explicitly to one of ``"torchcodec"``, ``"ffmpeg"``, or ``"pyav"``
    # to pin a single backend — e.g. ``"ffmpeg"`` to skip a torchcodec that
    # cannot decode the dataset's codec ("Operation not permitted").
    video_backend: str | None = None
    _meta: Any = field(default=None, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _camera_keys: list[str] = field(default_factory=list, init=False, repr=False)
    # Pipeline runs the three module phases under a ThreadPoolExecutor (see
    # ``ExecutorConfig.episode_parallelism``); guard the dict cache and the
    # one-shot warn flag against concurrent updates from worker threads.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415

        self._meta = LeRobotDatasetMetadata(repo_id="local", root=self.root)
        # ``camera_keys`` covers both image- and video-stored cameras and is
        # always defined on the metadata (``[]`` in the worst case), so it is
        # the single source we need here.
        keys = list(self._meta.camera_keys)
        # Last-resort fallback: if metadata didn't surface anything but the
        # caller explicitly named a camera (``--vlm.camera_key=...``), trust
        # them — the key is by definition known to exist on the dataset.
        if not keys and self.camera_key:
            keys = [self.camera_key]
        self._camera_keys = keys
        if self.camera_key is None:
            self.camera_key = keys[0] if keys else None

    @property
    def camera_keys(self) -> list[str]:
        """All ``observation.images.*`` keys available on this dataset."""
        return list(self._camera_keys)

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
    ) -> list[Any]:
        target = camera_key if camera_key is not None else self.camera_key
        if not timestamps or target is None:
            return []

        out: list[Any] = []
        misses: list[float] = []
        miss_indices: list[int] = []
        with self._lock:
            for i, ts in enumerate(timestamps):
                key = (record.episode_index, target, round(float(ts), 6))
                cached = self._cache.get(key)
                if cached is not None:
                    out.append(cached)
                else:
                    out.append(None)
                    misses.append(float(ts))
                    miss_indices.append(i)

        if misses:
            decoded = self._decode(record.episode_index, misses, target)
            # ``_decode`` returns exactly one frame per requested timestamp,
            # or an empty list if decoding failed wholesale. A partial list
            # would mean a frame/timestamp misalignment, so only pair them up
            # when the counts match (``strict=True`` then guards regressions).
            if len(decoded) == len(miss_indices):
                with self._lock:
                    for i, frame in zip(miss_indices, decoded, strict=True):
                        out[i] = frame
                        key = (record.episode_index, target, round(float(timestamps[i]), 6))
                        if len(self._cache) >= self.cache_size:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[key] = frame
        # filter out any None left over from decode failures
        return [frame for frame in out if frame is not None]

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        """Return up to ``max_frames`` frames uniformly sampled across the episode.

        The whole episode duration is covered; the model picks subtask
        boundaries from the temporal pooling it does internally. Frames are
        ``torch.Tensor`` (see :meth:`frames_at`).
        """
        target = camera_key if camera_key is not None else self.camera_key
        if max_frames <= 0 or target is None or not record.frame_timestamps:
            return []
        n_frames = min(max_frames, len(record.frame_timestamps))
        if n_frames == len(record.frame_timestamps):
            timestamps = list(record.frame_timestamps)
        else:
            t0 = record.frame_timestamps[0]
            t_last = record.frame_timestamps[-1]
            if t_last <= t0:
                timestamps = [float(t0)] * n_frames
            else:
                step = (t_last - t0) / (n_frames - 1) if n_frames > 1 else 0.0
                timestamps = [float(t0 + i * step) for i in range(n_frames)]
        return self.frames_at(record, timestamps, camera_key=target)

    def episode_clip_path(self, record: EpisodeRecord, cache_dir: Path) -> Path | None:
        """Extract the episode's subclip to ``cache_dir/ep_{idx:06d}.mp4``.

        Returns ``None`` if the dataset has no video tracks. Skips
        re-extract when the cached clip already exists. Re-encodes to
        H.264 (libx264) so the resulting mp4 is decodable by every
        downstream video processor — stream-copy would inherit the
        source codec (often AV1 in modern LeRobot datasets), which
        vllm's libav build cannot decode.
        """
        import subprocess  # noqa: PLC0415

        if self.camera_key is None:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"ep_{record.episode_index:06d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        ep = self._meta.episodes[record.episode_index]
        from_timestamp = float(ep[f"videos/{self.camera_key}/from_timestamp"])
        to_timestamp = float(ep[f"videos/{self.camera_key}/to_timestamp"])
        src = self.root / self._meta.get_video_file_path(record.episode_index, self.camera_key)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{from_timestamp:.3f}",
            "-to",
            f"{to_timestamp:.3f}",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None

    def _decode(self, episode_index: int, timestamps: list[float], camera_key: str) -> list[Any]:
        """Decode ``timestamps`` from the episode's video as ``(C, H, W)`` tensors.

        Delegates to :func:`lerobot.datasets.video_utils.decode_video_frames`
        (torchcodec by default, PyAV fallback) rather than a bespoke decoder.
        Returns one frame per requested timestamp, or ``[]`` if decoding
        failed wholesale — callers treat ``[]`` as "no frames available".
        """
        ep = self._meta.episodes[episode_index]
        from_timestamp = ep[f"videos/{camera_key}/from_timestamp"]
        shifted = [from_timestamp + ts for ts in timestamps]
        video_path = self.root / self._meta.get_video_file_path(episode_index, camera_key)

        # Build the decoder chain. In-process decoders are fragile here:
        # torchcodec raises in some containers (vllm-openai: "Operation not
        # permitted"), lerobot's ``pyav`` backend routes through
        # ``torchvision.io.VideoReader`` (removed in torchvision 0.23+), and
        # PyAV can outright SIGSEGV on the AV1 streams modern LeRobot
        # datasets use. ``_decode_frames_ffmpeg`` shells out to the ffmpeg
        # CLI — it decodes AV1 and a crash stays isolated to the child
        # process — so it is the always-available fallback.
        if self.video_backend:
            chain = [self.video_backend]
        else:
            chain = (["torchcodec"] if get_safe_default_codec() == "torchcodec" else []) + ["ffmpeg"]

        exc: Exception | None = None
        for backend in chain:
            try:
                if backend == "ffmpeg":
                    return _decode_frames_ffmpeg(video_path, shifted)
                if backend in ("pyav", "av"):
                    return _decode_frames_av(video_path, shifted)
                # Stacked ``(N, C, H, W)`` uint8 tensor; one row per timestamp.
                decoded = decode_video_frames(
                    video_path, shifted, self.tolerance_s, backend=backend, return_uint8=True
                )
                return list(decoded)
            except Exception as e:  # noqa: PERF203
                exc = e

        # Every backend raised. Log loudly the first time so a silent
        # vqa-module no-op (every prompt skipped because frames_at returned
        # []) is debuggable from the job log instead of post-hoc parquet
        # inspection. Subsequent failures stay quiet.
        with self._lock:
            already_warned = getattr(self, "_warned_decode_fail", False)
            if not already_warned:
                self._warned_decode_fail = True
        if not already_warned:
            logger.warning(
                "VideoFrameProvider._decode failed for episode=%s camera=%s "
                "video_path=%s backends=%s: %s",
                episode_index,
                camera_key,
                video_path,
                chain,
                exc,
                exc_info=exc,
            )
        return []


def make_frame_provider(
    root: Path, camera_key: str | None = None, video_backend: str | None = None
) -> FrameProvider:
    """Build a :class:`VideoFrameProvider` if videos are present, else null."""
    try:
        provider = VideoFrameProvider(root=root, camera_key=camera_key, video_backend=video_backend)
    except Exception:
        return null_provider()
    if provider.camera_key is None:
        return null_provider()
    return provider


def _decode_frames_ffmpeg(video_path: Path, timestamps: list[float]) -> list[Any]:
    """Decode the frames nearest to ``timestamps`` via the ffmpeg CLI.

    Runs one ``ffmpeg`` process per timestamp, seeking with ``-ss`` and
    piping a single PNG to stdout. Unlike the in-process decoders this
    survives a hostile container: a full ffmpeg build decodes AV1 (the codec
    modern LeRobot datasets use) where torchcodec raises and PyAV can
    SIGSEGV, and a crash stays isolated to the child process — a non-zero
    exit is a catchable error, not a segfault of the whole job. Returns one
    ``(C, H, W)`` uint8 tensor per timestamp.
    """
    import io  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    frames: list[Any] = []
    for ts in timestamps:
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error",
                "-ss", f"{max(ts, 0.0):.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-f", "image2pipe", "-vcodec", "png", "pipe:1",
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        if not proc.stdout:
            raise RuntimeError(f"ffmpeg returned no frame for t={ts:.3f}s of {video_path}")
        img = PIL.Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        frames.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous())
    return frames


def _decode_frames_av(video_path: Path, timestamps: list[float]) -> list[Any]:
    """Decode the frames nearest to ``timestamps`` using PyAV directly.

    lerobot's ``decode_video_frames(backend="pyav")`` routes through
    ``torchvision.io.VideoReader``, removed in torchvision 0.23+. This helper
    talks to the ``av`` package directly. Note PyAV can SIGSEGV on AV1
    streams in some builds — prefer ``_decode_frames_ffmpeg`` as the default
    fallback; this stays available behind ``video_backend="pyav"``. Returns
    one ``(C, H, W)`` uint8 tensor per timestamp.
    """
    import av  # noqa: PLC0415

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # Seek to the keyframe at or before the first requested timestamp.
        offset = max(int(first_ts / stream.time_base), 0) if stream.time_base else 0
        container.seek(offset, stream=stream, backward=True, any_frame=False)
        for idx, frame in enumerate(container.decode(stream)):
            ts = frame.time
            if ts is None:
                ts = float(frame.pts * stream.time_base) if frame.pts is not None else float(idx)
            loaded_ts.append(ts)
            loaded_frames.append(
                torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous()
            )
            if ts >= last_ts:
                break
    if not loaded_frames:
        raise RuntimeError(f"PyAV decoded no frames from {video_path}")
    ts_tensor = torch.tensor(loaded_ts)
    return [loaded_frames[int(torch.argmin((ts_tensor - q).abs()))] for q in timestamps]


def _frame_to_pil(frame: Any) -> Any:
    """Materialise a decoded frame as a ``PIL.Image`` for the VLM message.

    Frames flow through the provider as ``torch.Tensor`` (``C, H, W`` uint8,
    straight from :func:`decode_video_frames`); PIL is only created here, at
    the VLM-message boundary, because the chat backends expect PIL images /
    data URLs. Non-tensor inputs (e.g. test stubs) pass through untouched.
    """
    if not isinstance(frame, torch.Tensor):
        return frame
    array = frame.detach().cpu()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    return PIL.Image.fromarray(array.to(torch.uint8).numpy())


def to_image_blocks(frames: list[Any]) -> list[dict[str, Any]]:
    """Convert decoded frames to Qwen-VL-compatible image content blocks."""
    return [{"type": "image", "image": _frame_to_pil(frame)} for frame in frames]


def to_video_block(frames: list[Any]) -> list[dict[str, Any]]:
    """Wrap a list of decoded frames as one Qwen-VL video block.

    Returns ``[]`` when the list is empty, so the caller can splat the result
    into a content array without a separate emptiness check.
    """
    if not frames:
        return []
    return [{"type": "video", "video": [_frame_to_pil(frame) for frame in frames]}]


def to_video_url_block(url: str | None, fps: float = 2.0) -> list[dict[str, Any]]:
    """Wrap a video file URL as one ``video_url`` block.

    Used by the ``openai`` backend (transformers serve / vllm serve /
    ktransformers serve), where the server handles frame sampling.
    Returns ``[]`` when ``url`` is ``None`` so the caller can splat.
    """
    if not url:
        return []
    return [{"type": "video_url", "video_url": {"url": url}, "fps": fps}]
