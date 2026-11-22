# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/attention/gdn.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.norms import RMSNorm

DT_BIAS_INIT = -5.0
A_LOG_INIT = 0.0


class GatedDeltaNet(nn.Module):
    """Parameters shared by the Gated DeltaNet layers of both policies: RMS-normalized queries and keys, a per-token write strength (beta), a per-chunk decay of the recurrent state, and a normalized, gated output. The policies' subclasses run the recurrence over their own chunks."""

    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-05)
        self.k_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-05)
        self.beta_proj = nn.Linear(hidden_size, heads, bias=True)
        self.gate_proj = nn.Linear(hidden_size, heads, bias=True)
        self.A_log = nn.Parameter(torch.full((heads,), A_LOG_INIT, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.full((heads,), DT_BIAS_INIT, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True
        self.output_gate = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_norm = RMSNorm(self.dim, scale_factor=1.0, eps=1e-05, norm_dim=-2)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def reset_recurrence_parameters(self) -> None:
        """Set the decay parameters to their starting values."""
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        with torch.no_grad():
            self.A_log.fill_(A_LOG_INIT)
            self.dt_bias.fill_(DT_BIAS_INIT)

    def decay_from_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Return the state decay ``exp(-exp(A_log) * softplus(gate + dt_bias))`` of float32 gate logits [..., heads]."""
        return (-self.A_log.float().exp() * F.softplus(gate + self.dt_bias.float())).exp()
