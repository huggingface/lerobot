# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/deploy/memory/view_stream.py. Import paths are rewritten to the LeRobot package;
# the plain path; the CUDA graphs of an optimized stream are not vendored, and strip_view_latents comes from sana_robot/data/ar/teacher_forcing.py.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch


def strip_view_latents(view_latents) -> torch.Tensor:
    """Return the latents of several views [..., F, h_v, w_v] as one strip [..., F, sum(h_v * w_v)], every latent frame the concatenation of its flattened views."""
    return torch.cat([latents.flatten(-2) for latents in view_latents], dim=-1)


class ViewStream:
    """The latents of an episode's camera views, encoded while the frames arrive.

    Training encodes an episode once with the causal video VAE. The stream gives the same latents frame block by
    frame block: the first frame alone is the sink, and the frames observed while a chunk ran, encoded after
    everything before them, are the chunk's target latents. Views of one size share a call of the encoder.
    """

    def __init__(self, encode) -> None:
        """Build a stream.

        Args:
            encode: The video encoder of ``sana_robot.utils.vae.load_video_encoder``.
        """
        self.encode = encode
        self.streams = {}

    def start(self, frames) -> torch.Tensor:
        """Open the stream of a new episode with its first frame.

        Args:
            frames: The first frame of every view, a tensor [3, 1, H_v, W_v] in [-1, 1] per view.

        Returns:
            The sink latent [1, C, 1, 1, sum(h_v * w_v)].
        """
        self.streams = {}
        return self._encode(frames, self.streams)

    def encode_chunk(self, frames) -> torch.Tensor:
        """Encode the frames observed while a chunk ran, after everything the stream has seen.

        Args:
            frames: The chunk's video frames of every view, a tensor [3, 8 * f, H_v, W_v] in [-1, 1] per view.

        Returns:
            The chunk's target latents [1, C, f, 1, sum(h_v * w_v)].
        """
        return self._encode(frames, self.streams)

    def encode_observation(self, frames) -> torch.Tensor:
        """Encode one frame of every view alone, as the observation of a chunk; the stream stays as it is.

        Args:
            frames: The current frame of every view, a tensor [3, 1, H_v, W_v] in [-1, 1] per view.

        Returns:
            The observation latent [1, C, 1, 1, sum(h_v * w_v)].
        """
        return self._encode(frames, {}, keep=False)

    def save(self) -> dict:
        """Return a copy of the stream's state for ``load``."""
        return {size: [entry.clone() for entry in state] for (size, state) in self.streams.items()}

    def load(self, saved: dict) -> None:
        """Bring a saved state back into the stream's own buffers."""
        for size, state in saved.items():
            self._keep_state(self.streams, size, state)

    def _encode(self, frames, streams: dict, keep: bool = True) -> torch.Tensor:
        """Encode the frames of every view, the views of one size in one call, and return the views' latents side by side; ``keep`` stores the encoder's new state in ``streams``."""
        groups = {}
        for view, clip in enumerate(frames):
            groups.setdefault(tuple(clip.shape[-2:]), []).append(view)
        latents = [None] * len(frames)
        for size, views in groups.items():
            outputs = self._encode_step(torch.stack([frames[view] for view in views]), *streams.get(size, ()))
            encoded = outputs[0].clone()
            if keep:
                self._keep_state(streams, size, outputs[1:])
            for row, view in enumerate(views):
                latents[view] = encoded[row : row + 1]
        return strip_view_latents(latents).unsqueeze(3)

    def _encode_step(self, frames, *state) -> tuple:
        """Encode frames after a state of the encoder and return the latents and the new state, as tensors."""
        stream = list(state)
        latents = self.encode(frames, stream)
        return (latents, *stream)

    @staticmethod
    def _keep_state(streams: dict, size, state) -> None:
        """Copy an encoder state into the buffers ``streams`` holds for views of a size, which keep their addresses; new buffers when it holds none of that layout."""
        buffers = streams.get(size)
        if buffers and all(
            buffer.shape == entry.shape for (buffer, entry) in zip(buffers, state, strict=False)
        ):
            for buffer, entry in zip(buffers, state, strict=False):
                buffer.copy_(entry)
        else:
            streams[size] = [entry.clone() for entry in state]
