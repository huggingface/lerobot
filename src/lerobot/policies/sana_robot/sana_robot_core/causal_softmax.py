# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/attention/causal_softmax.py. Import paths are rewritten to the LeRobot package;
# the streaming window; attention over the teacher-forcing sequence is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.softmax import GatedSoftmaxAttention, rotate


class CausalSoftmaxAttention(GatedSoftmaxAttention):
    """Softmax attention of SANA-Robot-AR. A streaming window reads the keys and values that the layer's cache manager (the layer cache's ``"kv"`` entry) selects, rotated at the positions the manager assigns."""

    def forward(
        self,
        x,
        rotary_emb,
        kv_cache,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
    ):
        (q, k, v) = self.project(x)
        dtype = v.dtype
        manager = kv_cache["kv"]
        if observation_prefill:
            (keys, values, valid, phases) = manager.observe(k.detach(), v.detach(), q)
        elif save_kv_cache:
            (keys, values, valid, phases) = manager.commit(k.detach(), v.detach(), q)
        else:
            (keys, values, valid, phases) = manager.read(k, v, q)
        if phases is not None:
            query_phases = phases.narrow(2, keys.shape[1] - q.shape[1], q.shape[1])
            (q, keys) = (rotate(q, query_phases), rotate(keys, phases))
        if valid is None:
            out = self.attend(q, keys, values)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=valid
            ).transpose(1, 2)
        return self.gate_and_project(out, x, dtype)
