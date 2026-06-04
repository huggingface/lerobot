#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical-parity tests for the SDPA attention port.

``pi05`` / ``pi052`` replaced the per-layer call from
``modeling_gemma.eager_attention_forward`` with
``sdpa_attention_forward`` (PyTorch SDPA + GQA repeat). The forward
output must be bit-equivalent (within bf16 tolerance) on the masks
this model actually uses — block-bidirectional with an arbitrary
additive bias — otherwise we silently change training behaviour.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.gemma import modeling_gemma  # noqa: E402

from lerobot.policies.pi052.pi05_backbone import (  # noqa: E402
    make_att_2d_masks,
    sdpa_attention_forward,
)
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE  # noqa: E402


def _mock_self_attn(num_kv_groups: int, training: bool = False):
    """Bare module surface that both forwards read."""
    return SimpleNamespace(
        num_key_value_groups=num_kv_groups,
        training=training,
    )


def _build_inputs(
    bsize: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(bsize, num_heads, seq_len, head_dim, dtype=dtype, generator=g)
    k = torch.randn(bsize, num_kv_heads, seq_len, head_dim, dtype=dtype, generator=g)
    v = torch.randn(bsize, num_kv_heads, seq_len, head_dim, dtype=dtype, generator=g)
    return q, k, v


def _block_bidirectional_mask(
    bsize: int, seq_len: int, block_sizes: list[int], dtype: torch.dtype
) -> torch.Tensor:
    """Mimic ``_prepare_attention_masks_4d`` on a block layout that
    matches ``[images, language, suffix]`` from ``embed_prefix`` +
    ``embed_suffix``: every block bidirectional internally, later
    blocks visible to earlier ones via the cumulative-block rule.
    """
    assert sum(block_sizes) == seq_len
    att_marks = []
    for i, n in enumerate(block_sizes):
        att_marks += [1 if i > 0 else 0] + [0] * (n - 1)
    pad = torch.ones(bsize, seq_len, dtype=torch.bool)
    att = torch.tensor(att_marks, dtype=torch.bool)[None].expand(bsize, seq_len)
    att_2d = make_att_2d_masks(pad, att)
    bias = torch.where(
        att_2d[:, None, :, :],
        torch.zeros((), dtype=dtype),
        torch.tensor(OPENPI_ATTENTION_MASK_VALUE, dtype=dtype),
    )
    return bias


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    [
        (8, 1, 256),  # gemma_2b / paligemma config
        (8, 8, 64),   # MHA control (no GQA repeat)
    ],
)
def test_sdpa_parity_with_eager_block_bidirectional(num_heads, num_kv_heads, head_dim):
    """SDPA forward output matches the eager softmax(QK^T)@V on the
    block-bidirectional mask layout pi05 actually uses."""
    bsize, seq_len = 2, 13
    block_sizes = [4, 5, 4]  # images, language, suffix-style blocks
    dtype = torch.float32   # cpu math kernel — keep fp32 for tight tol
    scaling = head_dim ** -0.5

    q, k, v = _build_inputs(bsize, num_heads, num_kv_heads, seq_len, head_dim, dtype)
    mask = _block_bidirectional_mask(bsize, seq_len, block_sizes, dtype)

    module = _mock_self_attn(num_heads // num_kv_heads)

    out_eager, _ = modeling_gemma.eager_attention_forward(
        module, q, k, v, mask, scaling
    )
    out_sdpa, _ = sdpa_attention_forward(
        module, q, k, v, mask, scaling
    )
    assert out_eager.shape == out_sdpa.shape
    torch.testing.assert_close(out_sdpa, out_eager, atol=1e-5, rtol=1e-4)


def test_sdpa_parity_bf16():
    """bf16 path — looser tolerance, must still match eager."""
    bsize, num_heads, num_kv_heads, seq_len, head_dim = 2, 8, 1, 17, 256
    scaling = head_dim ** -0.5
    q, k, v = _build_inputs(bsize, num_heads, num_kv_heads, seq_len, head_dim, torch.bfloat16)
    mask = _block_bidirectional_mask(bsize, seq_len, [5, 6, 6], torch.bfloat16)
    module = _mock_self_attn(num_heads // num_kv_heads)

    out_eager, _ = modeling_gemma.eager_attention_forward(
        module, q, k, v, mask, scaling
    )
    out_sdpa, _ = sdpa_attention_forward(
        module, q, k, v, mask, scaling
    )
    torch.testing.assert_close(out_sdpa, out_eager, atol=2e-2, rtol=2e-2)


def test_sdpa_parity_backward():
    """Gradients flow through SDPA and match the eager path within
    bf16 tolerance — critical for any training-side parity claim."""
    bsize, num_heads, num_kv_heads, seq_len, head_dim = 1, 4, 2, 9, 32
    scaling = head_dim ** -0.5
    q, k, v = _build_inputs(bsize, num_heads, num_kv_heads, seq_len, head_dim, torch.float32)
    q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
    mask = _block_bidirectional_mask(bsize, seq_len, [3, 3, 3], torch.float32)
    module = _mock_self_attn(num_heads // num_kv_heads)

    out_e, _ = modeling_gemma.eager_attention_forward(module, q, k, v, mask, scaling)
    g_q_e, g_k_e, g_v_e = torch.autograd.grad(out_e.sum(), [q, k, v])

    out_s, _ = sdpa_attention_forward(module, q, k, v, mask, scaling)
    g_q_s, g_k_s, g_v_s = torch.autograd.grad(out_s.sum(), [q, k, v])

    torch.testing.assert_close(g_q_s, g_q_e, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(g_k_s, g_k_e, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(g_v_s, g_v_e, atol=1e-5, rtol=1e-4)
