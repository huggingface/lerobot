# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/data/ar/teacher_forcing.py. Import paths are rewritten to the LeRobot package;
# a window is encoded at its own length: the block-wise encoder is not compiled.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch

from lerobot.policies.sana_robot.sana_robot_core.vae import encode_in_batches
from lerobot.policies.sana_robot.sana_robot_core.video import VAE_TEMPORAL_STRIDE

ENCODE_CHUNK_BUCKET = 1
OBSERVATION_BATCH = 64


def bucket_padding_chunks(chunks: int) -> int:
    """Return the chunks that pad a window to a multiple of ``ENCODE_CHUNK_BUCKET`` chunks, so that the compiled video encoder meets few clip lengths."""
    return -int(chunks) % ENCODE_CHUNK_BUCKET


def teacher_forcing_frame_count(chunks: int, latents_per_chunk: int = 1) -> int:
    """Return the latent frames of the teacher-forcing sequence of a window: the sink, the clean targets of all chunks but the last, and per chunk its observation and its noisy targets."""
    return 1 + (int(chunks) - 1) * int(latents_per_chunk) + int(chunks) * (1 + int(latents_per_chunk))


def target_frame_positions(chunks: int, latents_per_chunk: int = 1) -> torch.Tensor:
    """Return the positions of the noisy target frames in the teacher-forcing sequence, chunk after chunk."""
    first = 1 + (int(chunks) - 1) * int(latents_per_chunk)
    offsets = torch.arange(int(chunks) * int(latents_per_chunk))
    return first + offsets + offsets // int(latents_per_chunk) + 1


def observation_frame_indices(chunks: int, frames_per_chunk: int) -> torch.Tensor:
    """Return the video frames that start chunks 1 to L - 1 of a window, the frames encoded alone as their observations."""
    return torch.arange(1, int(chunks)) * int(frames_per_chunk)


def encode_teacher_forcing_latents(
    encode, frames: torch.Tensor, chunks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the latents of one view of a batch of windows, frames [B, T, 3, H, W] in [-1, 1]: the causal encoding of the whole clip [B, C, 1 + L * f, h, w] and the first frame of chunks 1 to L - 1 encoded alone [B, C, L - 1, h, w]."""
    (batch, chunks) = (frames.shape[0], int(chunks))
    frames_per_chunk = (frames.shape[1] - 1) // chunks
    filler = frames[:, -1:].expand(-1, bucket_padding_chunks(chunks) * frames_per_chunk, -1, -1, -1)
    window_latents = encode(torch.cat((frames, filler), dim=1).transpose(1, 2))
    window_latents = window_latents[:, :, : 1 + chunks * (frames_per_chunk // VAE_TEMPORAL_STRIDE)]
    if chunks == 1:
        return window_latents, window_latents[:, :, :0]
    alone = (
        frames[:, observation_frame_indices(chunks, frames_per_chunk).to(frames.device)]
        .flatten(0, 1)
        .unsqueeze(2)
    )
    observation_latents = encode_in_batches(encode, alone, OBSERVATION_BATCH)
    return window_latents, observation_latents.squeeze(2).unflatten(0, (batch, chunks - 1)).transpose(1, 2)


def strip_view_latents(view_latents) -> torch.Tensor:
    """Return the latents of several views [..., F, h_v, w_v] as one strip [..., F, sum(h_v * w_v)], every latent frame the concatenation of its flattened views."""
    return torch.cat([latents.flatten(-2) for latents in view_latents], dim=-1)


def assemble_teacher_forcing_latents(
    window_latents: torch.Tensor, observation_latents: torch.Tensor, chunks: int
) -> torch.Tensor:
    """Return the clean teacher-forcing sequence [..., F, tokens] of a window from its causal encoding [..., 1 + L * f, tokens] and its observations [..., L - 1, tokens]: [sink | targets of chunks 0..L-2 | per chunk its observation and its targets], the sink being the first chunk's observation."""
    chunks = int(chunks)
    latents_per_chunk = (window_latents.shape[-2] - 1) // chunks
    sink = window_latents[..., :1, :]
    targets = window_latents[..., 1:, :]
    observations = torch.cat((sink, observation_latents), dim=-2)
    pieces = [sink, targets[..., : (chunks - 1) * latents_per_chunk, :]]
    for chunk in range(chunks):
        pieces.append(observations[..., chunk : chunk + 1, :])
        pieces.append(targets[..., chunk * latents_per_chunk : (chunk + 1) * latents_per_chunk, :])
    return torch.cat(pieces, dim=-2)


def teacher_forcing_video_timesteps(
    chunk_timesteps: torch.Tensor, latents_per_chunk: int = 1
) -> torch.Tensor:
    """Return the timestep of every latent frame of the teacher-forcing sequence [B, F] from one timestep per chunk [B, L]: a chunk's timestep on its noisy targets and 0 on the clean frames."""
    batch, chunks = chunk_timesteps.shape
    timesteps = chunk_timesteps.new_zeros(batch, teacher_forcing_frame_count(chunks, latents_per_chunk))
    positions = target_frame_positions(chunks, latents_per_chunk).to(chunk_timesteps.device)
    timesteps[:, positions] = chunk_timesteps.repeat_interleave(int(latents_per_chunk), dim=1)
    return timesteps


def action_row_positions(chunks: int, chunk_size: int) -> torch.Tensor:
    """Return where the L * C action rows of a window sit among its robot rows after the first state: every later chunk puts its state row before its actions."""
    rows = torch.arange(int(chunks) * int(chunk_size))
    return rows + rows // int(chunk_size)


def with_state_rows(action_rows: torch.Tensor, chunks: int, chunk_size: int) -> torch.Tensor:
    """Return per-action-row values [..., L * C, D] in the layout of the robot rows after the first state [..., L * (C + 1) - 1, D], zero on the rows of the later states."""
    shape = (*action_rows.shape[:-2], int(chunks) * (int(chunk_size) + 1) - 1, action_rows.shape[-1])
    rows = action_rows.new_zeros(shape)
    rows[..., action_row_positions(chunks, chunk_size).to(action_rows.device), :] = action_rows
    return rows


def teacher_forcing_action_timesteps(chunk_timesteps: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Return the timestep of every robot row after the first state [B, L * (C + 1) - 1] from one timestep per chunk [B, L]: a chunk's timestep on its action rows and 0 on the state rows."""
    chunks = chunk_timesteps.shape[1]
    per_row = chunk_timesteps.repeat_interleave(int(chunk_size), dim=1).unsqueeze(-1)
    return with_state_rows(per_row, chunks, chunk_size).squeeze(-1)
