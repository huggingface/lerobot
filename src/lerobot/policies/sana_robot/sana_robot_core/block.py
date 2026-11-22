# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/backbone/block.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.sana_robot.sana_robot_core.embedders import modulate
from lerobot.policies.sana_robot.sana_robot_core.swiglu import SwiGLU


def softmax_layer_indices(depth: int, softmax_ratio: float) -> list[int]:
    """Return the layers that use softmax attention, spread evenly over the depth."""
    softmax_layers = max(1, int(depth * softmax_ratio))
    step = depth / softmax_layers
    return [int((index + 1) * step) - 1 for index in range(softmax_layers)]


class PolicyBlock(nn.Module):
    """Transformer layer run as two residual sublayers: self-attention followed by text cross-attention, then the MLP. Every token carries its own AdaLN modulation."""

    def __init__(
        self, hidden_size: int, self_attention: nn.Module, cross_attention: nn.Module, mlp_ratio: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = self_attention
        self.cross_attn = cross_attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, *arguments, sublayer: str, **keyword_arguments):
        """Run the attention sublayer ("attention") or the MLP sublayer ("mlp") and return its residual update."""
        if sublayer == "attention":
            return self.attention_sublayer(*arguments, **keyword_arguments)
        return self.mlp_sublayer(*arguments, **keyword_arguments)

    def modulation_terms(self, modulation: torch.Tensor, hidden: torch.Tensor, first: int):
        """Return the (shift, scale, gate) of one sublayer from the per-token modulation [B, 1, N, 6C]; ``first`` is 0 for the attention sublayer and 3 for the MLP."""
        (batch, tokens, channels) = hidden.shape
        terms = (self.scale_shift_table[None, None] + modulation.reshape(batch, tokens, 6, channels)).unbind(
            2
        )
        return terms[first : first + 3]

    def attention_sublayer(self, hidden, text, modulation, text_arguments: dict, **attention_arguments):
        """Return the residual update of self-attention plus text cross-attention; cross-attention reads the tokens after the self-attention update."""
        (shift, scale, gate) = self.modulation_terms(modulation, hidden, 0)
        attention_delta = gate * self.attn(modulate(self.norm1(hidden), shift, scale), **attention_arguments)
        return attention_delta + self.cross_attn(hidden + attention_delta, text, **text_arguments)

    def mlp_sublayer(self, hidden, modulation):
        """Return the residual update of the MLP."""
        (shift, scale, gate) = self.modulation_terms(modulation, hidden, 3)
        return gate * self.mlp(modulate(self.norm2(hidden), shift, scale))
