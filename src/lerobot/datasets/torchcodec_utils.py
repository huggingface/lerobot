"""TorchCodec helpers for sparse MP4 IO with optional custom frame mappings."""

from __future__ import annotations

import json
from typing import Any

import torch
from torchcodec import FrameBatch, _core as core
from torchcodec.decoders._video_decoder import _get_and_validate_stream_metadata


def frame_mappings_tensors(payload: bytes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = json.loads(payload)
    frames = data["frames"]
    pts = torch.tensor([int(f["pts"]) for f in frames], dtype=torch.int64)
    key = torch.tensor([bool(f["key_frame"]) for f in frames], dtype=torch.bool)
    dur = torch.tensor([int(f["duration"]) for f in frames], dtype=torch.int64)
    return pts, key, dur


class VideoDecoderLike:
    """Minimal VideoDecoder surface used by episode byte cache."""

    def __init__(self, decoder: torch.Tensor, *, stream_index: int | None = None):
        self._decoder = decoder
        (
            self.metadata,
            self.stream_index,
            self._begin_stream_seconds,
            self._end_stream_seconds,
            self._num_frames,
        ) = _get_and_validate_stream_metadata(decoder=decoder, stream_index=stream_index)

    def get_frames_played_at(self, seconds: list[float]) -> FrameBatch:
        return FrameBatch(*core.get_frames_by_pts(self._decoder, timestamps=seconds))


def open_video_decoder(source: Any, *, frame_mappings: bytes | None = None) -> VideoDecoderLike:
    """Open a decoder on sparse or full MP4 IO, skipping metadata scan when mappings exist."""
    if frame_mappings is None:
        decoder = core.create_from_file_like(source, "approximate")
        core.add_video_stream(decoder)
        return VideoDecoderLike(decoder)

    mappings = frame_mappings_tensors(frame_mappings)
    decoder = core.create_from_file_like(source, "custom_frame_mappings")
    core.add_video_stream(decoder, custom_frame_mappings=mappings)
    return VideoDecoderLike(decoder)
