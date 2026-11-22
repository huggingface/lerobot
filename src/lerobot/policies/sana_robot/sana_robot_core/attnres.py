# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/backbone/attnres.py. Import paths are rewritten to the LeRobot package;
# the compiled and the buffered mixes of training and of the optimized deployment are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.norms import DepthRMSNorm


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
