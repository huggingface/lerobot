# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/data/format/video.py. Import paths are rewritten to the LeRobot package;
# the resize of frame tensors; the decoding of video files is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch

VIDEO_FRAME_STRIDE = 4
VAE_TEMPORAL_STRIDE = 8
VAE_SPATIAL_STRIDE = 32
RESIZE_STRETCH = "stretch"
RESIZE_CROP = "crop"


def byte_frames(video: torch.Tensor) -> torch.Tensor:
    """Return float frames in [0, 1] as uint8 frames, rounded to the nearest value."""
    return video.mul_(255).round_().clamp_(0, 255).to(torch.uint8)


def resize_frames(video: torch.Tensor, size, resize: str) -> torch.Tensor:
    """Return float frames [T, 3, H, W] at ``size`` = (h, w) by bilinear interpolation: stretched to it (``stretch``), or scaled until they cover it and cut at the centre (``crop``)."""
    height, width = video.shape[-2:]
    target_height, target_width = (int(side) for side in size)
    if (height, width) == (target_height, target_width):
        return video
    if resize == RESIZE_STRETCH:
        scaled = (target_height, target_width)
    elif target_height / height > target_width / width:
        scaled = (target_height, round(width * (target_height / height)))
    else:
        scaled = (round(height * (target_width / width)), target_width)
    video = torch.nn.functional.interpolate(video, size=scaled, mode="bilinear", align_corners=False)
    top = int(round(scaled[0] - target_height) / 2.0)
    left = int(round(scaled[1] - target_width) / 2.0)
    return video[..., top : top + target_height, left : left + target_width]


def frames_to_model_range(frames: torch.Tensor) -> torch.Tensor:
    """Return uint8 frames as float32 values in [-1, 1], the input range of the video VAE."""
    return frames.float().div_(127.5).sub_(1.0)
