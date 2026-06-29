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

import io
import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import PIL.Image
import torch

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.video_utils import decode_video_frames, reencode_video

from .reader import EpisodeRecord, snap_to_frame

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
    # Keyframe decode backend forwarded to
    # :func:`lerobot.datasets.video_utils.decode_video_frames`. ``None``
    # uses the library default (torchcodec when available, else PyAV).
    video_backend: str | None = None
    _meta: Any = field(default=None, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _camera_keys: list[str] = field(default_factory=list, init=False, repr=False)
    # Pipeline runs the three module phases under a ThreadPoolExecutor (see
    # ``ExecutorConfig.episode_parallelism``); guard the dict cache and the
    # one-shot warn flag against concurrent updates from worker threads.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # Serializes decode_video_frames calls: torchcodec hands out one
    # ``VideoDecoder`` per file from a process-wide cache, and the decoder
    # is not safe to drive from multiple threads at once.
    _decode_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _warned_decode_fail: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415

        self._meta = LeRobotDatasetMetadata(repo_id="local", root=self.root)
        # Only ``video_keys`` are decodable here: the clip/decode paths read
        # ``videos/<key>/from_timestamp`` from episode metadata, which exists
        # only for video-stored cameras. Image-stored cameras (also in
        # ``camera_keys``) would KeyError, so restrict the list — and the
        # default — to video keys.
        # Depth cameras are excluded from the annotation pipeline for now.
        depth_keys = set(self._meta.depth_keys)
        keys = [key for key in self._meta.video_keys if key not in depth_keys]
        # Last-resort fallback: if metadata didn't surface any video keys but
        # the caller explicitly named a camera (``--vlm.camera_key=...``),
        # trust them — the key is by definition known to exist on the dataset.
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
        # Snap each request to the nearest real frame timestamp: callers
        # sample uniform grids whose points land mid-frame, and
        # ``decode_video_frames`` rejects queries farther than
        # ``tolerance_s`` from a decodable frame. Snapping also dedupes
        # repeat queries through the cache.
        if record.frame_timestamps:
            timestamps = [snap_to_frame(float(ts), record.frame_timestamps) for ts in timestamps]

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

        Returns ``None`` if the dataset has no video tracks or extraction
        failed. Skips re-extract when the cached clip already exists.
        Re-encodes to H.264 via
        :func:`lerobot.datasets.video_utils.reencode_video` so the resulting
        mp4 is decodable by every downstream video processor — stream-copy
        would inherit the source codec (often AV1 in modern LeRobot
        datasets), which vllm's libav build cannot decode.
        """
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
        encoder = RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=None, crf=23, preset="ultrafast")
        try:
            reencode_video(
                src,
                out_path,
                video_encoder=encoder,
                overwrite=True,
                start_time_s=from_timestamp,
                end_time_s=to_timestamp,
            )
        except Exception:
            logger.warning(
                "clip extraction failed for episode %s (%s)", record.episode_index, src, exc_info=True
            )
            return None
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None

    def _decode(self, episode_index: int, timestamps: list[float], camera_key: str) -> list[Any]:
        """Decode ``timestamps`` from the episode's video as ``(C, H, W)`` tensors.

        Delegates to :func:`lerobot.datasets.video_utils.decode_video_frames`
        (torchcodec when available, PyAV otherwise; ``video_backend`` pins
        one explicitly). Returns one frame per requested timestamp, or ``[]``
        if decoding failed — callers treat ``[]`` as "no frames available".
        """
        ep = self._meta.episodes[episode_index]
        from_timestamp = ep[f"videos/{camera_key}/from_timestamp"]
        shifted = [from_timestamp + ts for ts in timestamps]
        video_path = self.root / self._meta.get_video_file_path(episode_index, camera_key)

        try:
            # The module phases decode under a ThreadPoolExecutor (see
            # ``ExecutorConfig.episode_parallelism``) but torchcodec's cached
            # per-file decoder is single-threaded, so serialize decodes on a
            # dedicated lock. Frame extraction is a small fraction of episode
            # wall time (VLM calls dominate), so the contention is cheap.
            with self._decode_lock:
                # Stacked ``(N, C, H, W)`` uint8 tensor; one row per timestamp.
                decoded = decode_video_frames(
                    video_path, shifted, self.tolerance_s, backend=self.video_backend, return_uint8=True
                )
            return list(decoded)
        except Exception as exc:
            # Log loudly the first time so a silent vqa-module no-op (every
            # prompt skipped because frames_at returned []) is debuggable from
            # the job log instead of post-hoc parquet inspection. Subsequent
            # failures stay quiet.
            with self._lock:
                already_warned = self._warned_decode_fail
                if not already_warned:
                    self._warned_decode_fail = True
            if not already_warned:
                logger.warning(
                    "VideoFrameProvider._decode failed for episode=%s camera=%s video_path=%s backend=%s: %s",
                    episode_index,
                    camera_key,
                    video_path,
                    self.video_backend,
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


def _draw_timestamp_badge(image: PIL.Image.Image, timestamp: float) -> PIL.Image.Image:
    """Burn ``timestamp`` (seconds) into the top-left corner of ``image``.

    A solid black badge with white text, so a VLM reading a contact sheet can
    cite the exact source time of each tile (e.g. ``012.50s``) directly,
    instead of the caller having to map tile position back to time. Mirrors
    the macrodata/refiner contact-sheet convention.
    """
    from PIL import ImageDraw, ImageFont

    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    label = f"{timestamp:06.2f}s"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = max(3, round(min(image.width, image.height) * 0.018))
    draw.rectangle((0, 0, text_w + pad * 2, text_h + pad * 2), fill=(0, 0, 0))
    draw.text((pad - left, pad - top), label, fill=(255, 255, 255), font=font)
    return result


def to_contact_sheet_blocks(
    frames: Sequence[Any],
    timestamps: Sequence[float],
    *,
    columns: int = 5,
    frames_per_sheet: int = 20,
    frame_width: int = 224,
    quality: int = 84,
) -> list[dict[str, Any]]:
    """Pack decoded frames into timestamped JPEG contact-sheet image blocks.

    Each frame is resized to ``frame_width`` wide, stamped with its
    episode-relative timestamp, and tiled row-major into grids of
    ``frames_per_sheet`` (``columns`` wide). One ``{"type":"image", ...}``
    block is returned per grid; many frames collapse into a few images, so a
    long episode's temporal coverage stays dense at a fraction of the vision
    tokens N separate frames would cost. ``frames`` and ``timestamps`` must be
    aligned and equal length. Returns ``[]`` for empty input.
    """
    from PIL import Image

    if not frames:
        return []
    columns = max(1, columns)
    frames_per_sheet = max(1, frames_per_sheet)
    rows_per_sheet = math.ceil(frames_per_sheet / columns)

    tiles: list[PIL.Image.Image] = []
    for ts, frame in zip(timestamps, frames, strict=False):
        img = _frame_to_pil(frame)
        if not isinstance(img, PIL.Image.Image):
            continue
        img = img.convert("RGB")
        if img.width != frame_width:
            height = max(1, round(img.height * frame_width / img.width))
            img = img.resize((frame_width, height), resample=Image.Resampling.BILINEAR)
        tiles.append(_draw_timestamp_badge(img, float(ts)))
    if not tiles:
        return []

    blocks: list[dict[str, Any]] = []
    for start in range(0, len(tiles), frames_per_sheet):
        chunk = tiles[start : start + frames_per_sheet]
        cell_w = max(tile.width for tile in chunk)
        cell_h = max(tile.height for tile in chunk)
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_per_sheet), color=(0, 0, 0))
        for i, tile in enumerate(chunk):
            x = (i % columns) * cell_w
            y = (i // columns) * cell_h
            sheet.paste(tile, (x, y))
        # JPEG round-trip at ``quality`` to match the refiner convention and
        # shrink the wire payload; vision-token count is set by resolution, so
        # the real saving is the grid packing, not the codec.
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        blocks.append({"type": "image", "image": Image.open(buf).convert("RGB")})
    return blocks
