# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVlabs/LongLive @ 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9,
# Long-WAM/src/longwam/runtime/streaming_vae.py (Apache-2.0).
# Original contributor source: Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1.
# Changes: typed native LeRobot interface; window and reset semantics preserved.
"""Window-scoped causal VAE encoding for either Long-WAM denoising mode."""

from typing import Any

import torch
from torch import Tensor


class StreamingVAE:
    """Encode observations incrementally, restarting causal padding each window."""

    def __init__(self, model: Any, *, past_steps: int, sample_stride: int) -> None:
        self.model = model
        self.past_steps, self.sample_stride = int(past_steps), int(sample_stride)
        self.group = int(model.vae.temporal_downsample_factor)
        if self.past_steps < 0 or self.sample_stride <= 0 or self.group <= 0:
            raise ValueError("Invalid VAE observation cadence")
        if self.past_steps % (self.sample_stride * self.group):
            raise ValueError("Streaming history must contain complete temporal VAE groups")
        self.anchor: int | None = None
        self._pending: list[Tensor] = []
        self._count = 0

    def reset(self) -> None:
        self.model.vae.model.clear_cache()
        self.anchor = None
        self._pending.clear()
        self._count = 0

    def begin(self, anchor: int, history: list[tuple[int, Tensor]]) -> None:
        self.reset()
        if not history:
            raise ValueError("A streaming window requires at least one observation")
        self.anchor = int(anchor)
        first = self.anchor - self.past_steps
        frames = dict(history)
        earliest, latest = min(frames), max(frames)
        for step in range(first, min(latest, self.anchor) + 1, self.sample_stride):
            if step < 0:
                frame = frames[earliest]
            else:
                if step not in frames:
                    raise ValueError(f"Missing observed frame {step} in the streaming window")
                frame = frames[step]
            self.feed(step, frame)

    @torch.no_grad()
    def feed(self, step: int, frame: Tensor) -> None:
        if self.anchor is None:
            return
        expected = self.anchor - self.past_steps + self._count * self.sample_stride
        if step != expected:
            if step > expected:
                raise ValueError(f"Skipped sampled observation {expected}")
            return
        if step > self.anchor:
            return
        self._count += 1
        self._pending.append(frame)
        count = 1 if self._count == 1 else self.group
        if len(self._pending) == count:
            chunk = torch.stack(self._pending, dim=1).unsqueeze(0)
            if self._count == 1:
                self.model._stream_video_latents_start(chunk)
            else:
                self.model._stream_video_latents_step(chunk)
            self._pending.clear()

    @torch.no_grad()
    def finish(self, anchor: int) -> Tensor:
        expected = self.past_steps // self.sample_stride + 1
        if anchor != self.anchor or self._count != expected or self._pending:
            raise RuntimeError("Streaming VAE window is incomplete or has the wrong anchor")
        result = self.model._stream_video_latents_finish()
        self.anchor = None
        if result.ndim != 5 or result.shape[2] != self.model.num_clean_frames:
            raise ValueError("Streaming VAE latent window differs from the model contract")
        return result
