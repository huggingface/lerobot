# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/layout/teacher_forcing.py. Import paths are rewritten to the LeRobot package;
# the packed row of several windows is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from collections.abc import Mapping

import torch

SINK_HORIZON_CHUNKS = 7


def resolve_actions_per_chunk(actions_per_chunk: int | Mapping[str, int], fps: float) -> int:
    """Return the control steps of one action chunk at the control frequency ``fps``."""
    if isinstance(actions_per_chunk, Mapping):
        return int(actions_per_chunk[str(int(round(float(fps))))])
    return int(actions_per_chunk)


def sink_read_chunks(num_chunks: int, window: int, device=None) -> torch.Tensor:
    """Return, per softmax window start, the chunk at whose start time the readers of that window see the sink. A reader in chunk c sees the sink at the start of chunk max(0, c - ``SINK_HORIZON_CHUNKS``), so the sink is never farther than the horizon in the past."""
    starts = torch.arange(int(num_chunks), device=device)
    return (starts + int(window) - SINK_HORIZON_CHUNKS).clamp_min(0)


def latent_centre_offset(vae_stride: int) -> float:
    """Return the middle of the ``vae_stride`` video frames that one target latent covers, in latent frames after the first of them: 0.5625 for a stride of 8."""
    return (int(vae_stride) + 1) / (2 * int(vae_stride))


def teacher_forcing_frame_positions(
    num_chunks: int, chunk_frames: int, vae_stride: int = 8, device=None
) -> torch.Tensor:
    """Return the time of every latent frame of the teacher-forcing sequence [sink | clean targets | noisy chunks], in latent frames. The sink sits at 0, an observation latent at the start of its chunk and a target latent at the middle of the video frames it covers."""
    chunk_frames = int(chunk_frames)
    target_offset = latent_centre_offset(vae_stride)
    positions = [0.0] + [frame + target_offset for frame in range((int(num_chunks) - 1) * chunk_frames)]
    for chunk in range(int(num_chunks)):
        positions.append(float(chunk * chunk_frames))
        positions.extend(chunk * chunk_frames + frame + target_offset for frame in range(chunk_frames))
    return torch.tensor(positions, dtype=torch.float64, device=device)


def noisy_chunk_frames(chunk_frames: int) -> int:
    """Return the latent frames of one noisy chunk of the teacher-forcing sequence: its observation latent, then its targets."""
    return 1 + int(chunk_frames)


def noisy_observation_spans(view_areas, chunk_frames: int) -> tuple:
    """Return the (offset, length) runs of the observation tokens inside a noisy chunk's segment [view 1; ...; view V; state; actions]: the first latent frame of every view and the state row."""
    (spans, offset) = ([], 0)
    for area in view_areas:
        spans.append((offset, int(area)))
        offset += (1 + int(chunk_frames)) * int(area)
    spans.append((offset, 1))
    return tuple(spans)


def teacher_forcing_observation_index(bounds, clean_segments: int, observation_spans) -> tuple:
    """Return the token index of the observation tokens of every noisy chunk and their per-chunk bounds."""
    bounds = [int(bound) for bound in bounds]
    tokens_per_chunk = sum(int(length) for (_offset, length) in observation_spans)
    parts = []
    for segment in range(int(clean_segments), len(bounds) - 1):
        for offset, length in observation_spans:
            parts.append(
                torch.arange(bounds[segment] + int(offset), bounds[segment] + int(offset) + int(length))
            )
    index = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
    chunks = len(bounds) - 1 - int(clean_segments)
    return (index, tuple(tokens_per_chunk * chunk for chunk in range(chunks + 1)))


def teacher_forcing_prediction_index(bounds, clean_segments: int, observation_spans) -> tuple:
    """Return the token index of the prediction tokens of every noisy chunk (its target latents and action rows) and their per-chunk bounds."""
    bounds = [int(bound) for bound in bounds]
    (observation, _) = teacher_forcing_observation_index(bounds, clean_segments, observation_spans)
    is_observation = torch.zeros(bounds[-1], dtype=torch.bool)
    is_observation[observation] = True
    (parts, offsets) = ([], [0])
    for segment in range(int(clean_segments), len(bounds) - 1):
        tokens = torch.arange(bounds[segment], bounds[segment + 1])
        parts.append(tokens[~is_observation[tokens]])
        offsets.append(offsets[-1] + int(parts[-1].numel()))
    index = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
    return (index, tuple(offsets))
