# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/attention/cross_attention.py. Import paths are rewritten to the LeRobot package;
# the parameters; the attention calls of SANA-Robot-base are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch.nn as nn

from lerobot.policies.sana_robot.sana_robot_core.norms import RMSNorm


class MultiHeadCrossAttention(nn.Module):
    """Parameters of the text cross-attention: RMS-normalized queries and keys. ``ChunkTextCrossAttention`` runs it, every token span on its own text group."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-06)
        self.k_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-06)
