# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/optim/model.py
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Deployment helpers extracted from the original model without changing weights."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .mot import MoT
    from .wan_video_dit import WanVideoDiT


class InferenceModelMixin:
    mot: MoT
    num_clean_frames: int
    video_expert: WanVideoDiT

    def _bucket_cached_inference_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact cached text without truncating visible context semantics."""
        bucket = getattr(self, "_context_token_bucket", None)
        if bucket is None:
            return context, context_mask
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "Context bucketing requires [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        bucket = int(bucket)
        if context.shape[1] <= bucket:
            return context, context_mask
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                "Context bucketing requires matching batch/token dimensions, got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if not bool(context_mask[:, bucket:].any().item()):
            return (
                context[:, :bucket].contiguous(),
                context_mask[:, :bucket].contiguous(),
            )

        if not bool(context_mask.all().item()):
            raise ValueError(
                f"Context has valid tokens beyond configured bucket {bucket}; "
                "increase context_token_bucket instead of truncating them."
            )
        zero_rows = context.eq(0).all(dim=-1)
        compact_bias = torch.zeros(
            (context.shape[0], bucket),
            dtype=context.dtype,
            device=context.device,
        )
        removed_slots = context.shape[1] - bucket
        for batch_index in range(context.shape[0]):
            zero_indices = torch.nonzero(
                zero_rows[batch_index, :bucket],
                as_tuple=False,
            ).flatten()
            if zero_indices.numel() == 0:
                raise ValueError(
                    f"Context has no repeated-zero slot inside bucket {bucket}; "
                    "increase context_token_bucket."
                )
            first_zero = int(zero_indices[0].item())
            if not bool(zero_rows[batch_index, first_zero:].all().item()):
                raise ValueError("Context zero padding must be a contiguous trailing suffix.")
            compact_bias[batch_index, first_zero] = math.log(removed_slots + 1)
        return context[:, :bucket].contiguous(), compact_bias

    @torch.no_grad()
    def _stream_video_latents_start(self, video_chunk):
        self.vae.stream_encode_start(video_chunk, device=self.device)

    @torch.no_grad()
    def _stream_video_latents_step(self, video_chunk):
        self.vae.stream_encode_step(video_chunk, device=self.device)

    @torch.no_grad()
    def _stream_video_latents_finish(self):
        return self.vae.stream_encode_finish()

    @torch.no_grad()
    def _stage_action_video_kv_cache(
        self,
        video_kv_cache: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Install the optional resident cache and remove K/V from graph inputs."""
        if not bool(getattr(self, "_action_video_kv_resident_cache_enabled", False)):
            return video_kv_cache
        if not self.mot.owns_action_video_kv_cache(video_kv_cache):
            self.mot.install_action_video_kv_cache(video_kv_cache)
        # _predict_action_noise_with_cache reads the module buffers.  Passing an
        # empty tree prevents SafeCompiledCallable/Inductor from staging the 60
        # leaves again on every denoise step.
        return []

    @torch.no_grad()
    def _prefill_video_clean_prefix(
        self,
        clean_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        use_precomputed_video_cross_kv: bool = False,
    ) -> list[Any]:
        """Evaluate the causally closed clean segment once and return layer K/V."""
        clean_frames = int(clean_latents.shape[2])
        if clean_frames != int(self.num_clean_frames):
            raise ValueError(f"clean-prefix frame count changed: {clean_frames} != {self.num_clean_frames}")
        timestep = torch.zeros(
            (int(clean_latents.shape[0]),),
            dtype=clean_latents.dtype,
            device=clean_latents.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=clean_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_clean_frames=clean_frames,
            context_is_projected=use_precomputed_video_cross_kv,
        )
        tokens = video_pre["tokens"]
        attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(tokens.shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=tokens.device,
        )
        return self.mot.prefill_video_clean_prefix_cache(
            video_tokens=tokens,
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
                "use_cached_kv": use_precomputed_video_cross_kv,
            },
            video_attention_mask=attention_mask,
        )

    @torch.no_grad()
    def _predict_video_noise_with_clean_prefix(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        num_clean_frames: int,
        use_precomputed_video_cross_kv: bool = False,
    ) -> torch.Tensor:
        """Predict future velocity with segment-local W4A4 and retained clean K/V."""
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_clean_frames=num_clean_frames,
            context_is_projected=use_precomputed_video_cross_kv,
        )
        tokens = video_pre["tokens"]
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        clean_tokens = int(num_clean_frames) * tokens_per_frame
        if not 0 < clean_tokens < int(tokens.shape[1]):
            raise ValueError(f"invalid clean-prefix token split {clean_tokens}/{tokens.shape[1]}")
        attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(tokens.shape[1]),
            video_tokens_per_frame=tokens_per_frame,
            device=tokens.device,
        )
        future_tokens = self.mot.forward_video_future_with_clean_prefix(
            video_tokens=tokens,
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
                "use_cached_kv": use_precomputed_video_cross_kv,
            },
            video_attention_mask=attention_mask,
        )
        future_t = video_pre["t"][:, clean_tokens:]
        output_tokens = self.video_expert.head(future_tokens, future_t)
        full_frames, grid_h, grid_w = video_pre["meta"]["grid_size"]
        future_frames = int(full_frames) - int(num_clean_frames)
        return self.video_expert.unpatchify(
            output_tokens,
            (future_frames, grid_h, grid_w),
        )
