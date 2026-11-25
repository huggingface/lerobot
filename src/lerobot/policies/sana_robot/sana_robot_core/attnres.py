# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/backbone/attnres.py. Import paths are rewritten to the LeRobot package;
# the compiled and the buffered mixes of the kernels and of the optimized deployment are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.norms import DepthRMSNorm


def _weighted_source_sum(weights: torch.Tensor, *sources: torch.Tensor) -> torch.Tensor:
    """Return the sum of the sources weighted per token, accumulated in float32 and cast to the first source's dtype."""
    mixed = weights[0].unsqueeze(-1) * sources[0]
    for weight, source in zip(weights[1:], sources[1:], strict=False):
        mixed = mixed + weight.unsqueeze(-1) * source
    return mixed.to(sources[0].dtype)


class BlockAttnRes(nn.Module):
    """Attention residuals over blocks of consecutive layers. The input of every sublayer is a per-token softmax mix of the embedding, the finished blocks and the running sum of the current block."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, 1, bias=False)
        self.mlp_proj = nn.Linear(hidden_size, 1, bias=False)
        self.final_proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.attn_proj.weight)
        nn.init.zeros_(self.mlp_proj.weight)
        nn.init.zeros_(self.final_proj.weight)
        self.key_norm = DepthRMSNorm(eps=1e-06)

    def attend(
        self, projection: nn.Linear, sources: list, partial_block: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mix the sources, and the running block when given, with autograd."""
        sources = sources + [partial_block] if partial_block is not None else sources
        if len(sources) == 1:
            return sources[0]
        values = torch.stack(sources, dim=0)
        logits = torch.einsum("d,nbtd->nbt", projection.weight.squeeze(), self.key_norm(values))
        return torch.einsum("nbt,nbtd->btd", F.softmax(logits, dim=0), values)

    def source_logit(self, projection: nn.Linear, source: torch.Tensor) -> torch.Tensor:
        """Return the [B, T] mixing logits of one source in float32 or wider, without storing its normalized copy."""
        dtype = torch.promote_types(source.dtype, torch.float32)
        with torch.autocast(device_type=source.device.type, enabled=False):
            source = source.to(dtype)
            scale = torch.rsqrt(
                torch.linalg.vector_norm(source, dim=-1).square() / source.shape[-1] + self.key_norm.eps
            )
            return scale * (source @ projection.weight.squeeze().to(dtype))

    def attend_cached(
        self,
        projection: nn.Linear,
        sources: list,
        source_logits: list,
        partial_block: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``attend`` with the logits of the finished sources computed once per forward; only the running block's logit is new."""
        sources = sources + [partial_block] if partial_block is not None else sources
        if len(sources) == 1:
            return sources[0]
        logits = list(source_logits)
        if partial_block is not None:
            logits.append(self.source_logit(projection, partial_block))
        weights = F.softmax(torch.stack(logits, dim=0), dim=0)
        return _weighted_source_sum(weights, *sources)
