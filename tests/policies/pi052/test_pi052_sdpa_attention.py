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
from torch import nn
from torch.utils.checkpoint import checkpoint

pytest.importorskip("transformers")

from transformers.models.gemma import modeling_gemma  # noqa: E402

from lerobot.policies.pi052.modeling_pi052 import make_att_2d_masks  # noqa: E402
from lerobot.policies.pi_gemma import sdpa_attention_forward  # noqa: E402
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
        (8, 8, 64),  # MHA control (no GQA repeat)
    ],
)
def test_sdpa_parity_with_eager_block_bidirectional(num_heads, num_kv_heads, head_dim):
    """SDPA forward output matches the eager softmax(QK^T)@V on the
    block-bidirectional mask layout pi05 actually uses."""
    bsize, seq_len = 2, 13
    block_sizes = [4, 5, 4]  # images, language, suffix-style blocks
    dtype = torch.float32  # cpu math kernel — keep fp32 for tight tol
    scaling = head_dim**-0.5

    q, k, v = _build_inputs(bsize, num_heads, num_kv_heads, seq_len, head_dim, dtype)
    mask = _block_bidirectional_mask(bsize, seq_len, block_sizes, dtype)

    module = _mock_self_attn(num_heads // num_kv_heads)

    out_eager, _ = modeling_gemma.eager_attention_forward(module, q, k, v, mask, scaling)
    out_sdpa, _ = sdpa_attention_forward(module, q, k, v, mask, scaling)
    assert out_eager.shape == out_sdpa.shape
    torch.testing.assert_close(out_sdpa, out_eager, atol=1e-5, rtol=1e-4)


def test_sdpa_parity_bf16():
    """bf16 path — looser tolerance, must still match eager."""
    bsize, num_heads, num_kv_heads, seq_len, head_dim = 2, 8, 1, 17, 256
    scaling = head_dim**-0.5
    q, k, v = _build_inputs(bsize, num_heads, num_kv_heads, seq_len, head_dim, torch.bfloat16)
    mask = _block_bidirectional_mask(bsize, seq_len, [5, 6, 6], torch.bfloat16)
    module = _mock_self_attn(num_heads // num_kv_heads)

    out_eager, _ = modeling_gemma.eager_attention_forward(module, q, k, v, mask, scaling)
    out_sdpa, _ = sdpa_attention_forward(module, q, k, v, mask, scaling)
    torch.testing.assert_close(out_sdpa, out_eager, atol=2e-2, rtol=2e-2)


def test_sdpa_parity_backward():
    """Gradients flow through SDPA and match the eager path within
    bf16 tolerance — critical for any training-side parity claim."""
    bsize, num_heads, num_kv_heads, seq_len, head_dim = 1, 4, 2, 9, 32
    scaling = head_dim**-0.5
    q, k, v = _build_inputs(bsize, num_heads, num_kv_heads, seq_len, head_dim, torch.float32)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    mask = _block_bidirectional_mask(bsize, seq_len, [3, 3, 3], torch.float32)
    module = _mock_self_attn(num_heads // num_kv_heads)

    out_e, _ = modeling_gemma.eager_attention_forward(module, q, k, v, mask, scaling)
    g_q_e, g_k_e, g_v_e = torch.autograd.grad(out_e.sum(), [q, k, v])

    out_s, _ = sdpa_attention_forward(module, q, k, v, mask, scaling)
    g_q_s, g_k_s, g_v_s = torch.autograd.grad(out_s.sum(), [q, k, v])

    torch.testing.assert_close(g_q_s, g_q_e, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(g_k_s, g_k_e, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(g_v_s, g_v_e, atol=1e-5, rtol=1e-4)


def test_bf16_large_scores_backward_matches_fp32():
    """BF16 score rounding must not turn separated logits into an artificial tie."""
    module = _mock_self_attn(1, training=True)
    q = torch.tensor([[[[128.0, 128.0]]]], dtype=torch.bfloat16, requires_grad=True)
    k = torch.tensor([[[[128.0, 0.0], [128.0, 0.125]]]], dtype=torch.bfloat16, requires_grad=True)
    v = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]], dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16)
    reference_inputs = [x.detach().float().requires_grad_() for x in (q, k, v)]
    rq, rk, rv = reference_inputs
    # Independent FP32 reference: scores 16384 and 16400, not two rounded 16384s.
    reference = ((rq @ rk.transpose(-1, -2)).softmax(-1) @ rv).transpose(1, 2)
    expected_grads = torch.autograd.grad(reference.sum(), reference_inputs)
    output, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=1.0)
    gradients = torch.autograd.grad(output.sum(), (q, k, v))
    torch.testing.assert_close(output.float(), reference, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(gradients, expected_grads, strict=True):
        assert torch.isfinite(actual).all()
        # Fused BF16 backward can leave ~3e-5 cancellation residuals near p=1;
        # eager BF16 score materialization instead produces gradients of order 64.
        torch.testing.assert_close(actual.float(), expected, atol=1e-4, rtol=0.02)


@pytest.mark.parametrize("use_checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_layer_defaults_to_sdpa_and_keeps_prefix_gradients(monkeypatch, use_checkpointing, dtype):
    """Exercise the real shared PI05/PI052 joint layer, including KI-off backprop."""
    from transformers.models.gemma.configuration_gemma import GemmaConfig

    from lerobot.policies.pi05 import modeling_pi05
    from lerobot.policies.pi_gemma import _get_pi_gemma_decoder_layer_base

    torch.manual_seed(14)
    config = GemmaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
    )
    layer_class = _get_pi_gemma_decoder_layer_base()
    prefix_layer = layer_class(config, 0)
    config.use_adarms = True
    config.adarms_cond_dim = 32
    action_layer = layer_class(config, 0)
    for norm in (action_layer.input_layernorm, action_layer.post_attention_layernorm):
        nn.init.normal_(norm.dense.weight, std=0.01)
    layers = nn.ModuleList([prefix_layer, action_layer]).to(dtype)
    for layer in layers:
        layer.input_layernorm.float()
        layer.post_attention_layernorm.float()
    rotary = modeling_gemma.GemmaRotaryEmbedding(config)
    prefix = torch.randn(2, 3, 32, dtype=dtype, requires_grad=True)
    suffix = torch.randn(2, 2, 32, dtype=dtype, requires_grad=True)
    cond = torch.randn(2, 32, requires_grad=True)
    mask = _block_bidirectional_mask(2, 5, [3, 2], torch.float32)
    positions = torch.arange(5)[None].expand(2, -1)
    calls = []

    def counted_sdpa(*args, **kwargs):
        assert all(x.dtype == torch.float32 for x in args[1:4])
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        calls.append(True)
        return sdpa_attention_forward(*args, **kwargs)

    monkeypatch.setattr(modeling_pi05, "sdpa_attention_forward", counted_sdpa)
    args = ([prefix, suffix], mask, positions, [None, cond])
    kwargs = {"layers": layers, "rotary_emb": rotary}
    if use_checkpointing:
        outputs = checkpoint(modeling_pi05.compute_layer_complete, *args, use_reentrant=False, **kwargs)
    else:
        outputs = modeling_pi05.compute_layer_complete(*args, **kwargs)
    assert outputs[1].shape == suffix.shape
    assert outputs[1].dtype == dtype
    outputs[1].square().mean().backward()
    assert calls, "The joint layer must not fall back to BF16 eager attention"
    for tensor in (prefix, suffix, cond):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert prefix_layer.self_attn.k_proj.weight.grad.abs().sum() > 0
    assert prefix_layer.self_attn.v_proj.weight.grad.abs().sum() > 0
