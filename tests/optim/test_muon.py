# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Unit tests for the vendored Muon optimizer.

The CPU tests exercise only the local (non-DTensor) code paths, which is exactly
what plain DDP / single-process training uses: Newton-Schulz runs on the full
gradient. The FSDP2 mega-batch paths require a distributed context and are
covered by the multi-GPU suite instead.
"""

import pytest
import torch
from torch import nn

pytest.importorskip("torch")

from lerobot.optim.muon import (  # noqa: E402
    DistributedMuon,
    _DEFAULT_ADAMW_NAME_PATTERNS,
    _is_muon_eligible_ndim,
    batched_newton_schulz,
    split_muon_adamw_params,
)


class _Experts(nn.Module):
    """Fused MoE-style 3D parameter with an upstream-compatible FQN."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 16, 16) * 0.02)


class _TinyLingBot(nn.Module):
    """2D linears, a 3D MoE-style stack, 1D biases and an lm_head-shaped weight."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.Linear(16, 16),
                        "mlp": nn.ModuleDict({"experts": _Experts()}),
                    }
                )
                for _ in range(2)
            ]
        )
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, idx):
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer["attn"](x)
            experts = layer["mlp"]["experts"].weight  # [E, M, H]
            x = x + torch.einsum("btm,emh->bth", x, experts).mean(dim=0)[:, :16]
        return self.lm_head(x)


def test_split_routes_embedding_lm_head_and_1d_to_adamw():
    model = _TinyLingBot()
    names = dict(model.named_parameters())
    # 2D/3D non-embedding weights are Muon-eligible.
    assert _is_muon_eligible_ndim(names["layers.0.attn.weight"])
    assert _is_muon_eligible_ndim(names["layers.0.mlp.experts.weight"])
    assert not _is_muon_eligible_ndim(names["layers.0.attn.bias"])  # 1D
    for pat in ("embed_tokens", "lm_head"):
        assert pat in _DEFAULT_ADAMW_NAME_PATTERNS

    # The public split API routes every trainable parameter into exactly one bucket.
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(model)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for p in muon_params + adamw_params} == trainable
    assert not {id(p) for p in muon_params} & {id(p) for p in adamw_params}

    # Embedding + lm_head + biases are on AdamW; linears and the 3D stack on Muon.
    for name in adamw_names:
        assert "embed" in name or "lm_head" in name or name.endswith("bias"), name
    assert any("experts" in name for name in muon_names)
    assert any(name.endswith("attn.weight") for name in muon_names)


def test_distributed_muon_rejects_1d_params():
    bias = nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="2D and 3D"):
        DistributedMuon([bias], lr=1e-3)


def test_newton_schulz_preserves_shape_dtype_and_orthogonalizes():
    torch.manual_seed(0)
    g = torch.randn(5, 64, 32)
    out = batched_newton_schulz(g, ns_steps=5)
    assert out.shape == g.shape
    assert out.dtype == g.dtype
    # Each output slice should have spectral-ish norm near 1 (quintic NS on the
    # normalized matrix keeps RMS close to the input's).
    norms = out.norm(dim=(-2, -1))
    assert torch.all(norms > 0.05)
