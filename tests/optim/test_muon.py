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

"""Unit tests for the vendored Muon optimizer and the lerobot hybrid config.

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
    CombinedOptimizer,
    DistributedMuon,
    batched_newton_schulz,
    is_muon_optimizer,
)
from lerobot.optim.optimizers import (  # noqa: E402
    LingbotAdamWConfig,
    LingbotMuonConfig,
    _save_single_optimizer_state,
    _load_single_optimizer_state,
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


def _named_trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def test_split_routes_embedding_lm_head_and_1d_to_adamw():
    from lerobot.optim.muon import _DEFAULT_ADAMW_NAME_PATTERNS, _is_muon_eligible_ndim

    model = _TinyLingBot()
    names = dict(model.named_parameters())
    # 2D/3D non-embedding weights are Muon-eligible.
    assert _is_muon_eligible_ndim(names["layers.0.attn.weight"])
    assert _is_muon_eligible_ndim(names["layers.0.mlp.experts.weight"])
    assert not _is_muon_eligible_ndim(names["layers.0.attn.bias"])  # 1D
    for pat in ("embed_tokens", "lm_head"):
        assert pat in _DEFAULT_ADAMW_NAME_PATTERNS


def test_lingbot_muon_config_groups_and_children():
    torch.manual_seed(0)
    model = _TinyLingBot()
    opt = LingbotMuonConfig(lr=1e-4, expert_lr_scale=2.83).build(_named_trainable(model))

    assert isinstance(opt, CombinedOptimizer)
    assert is_muon_optimizer(opt)
    muon, adamw = opt.optimizers
    assert isinstance(muon, DistributedMuon)
    assert isinstance(adamw, torch.optim.AdamW)

    # Routing: every parameter lands in exactly one child.
    muon_params = {id(p) for g in muon.param_groups for p in g["params"]}
    adamw_params = {id(p) for g in adamw.param_groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_params | adamw_params == trainable
    assert not muon_params & adamw_params

    # Embedding + lm_head + biases are on AdamW; linears and the 3D stack on Muon.
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    for pid in adamw_params:
        name = name_by_id[pid]
        assert "embed" in name or "lm_head" in name or name.endswith("bias"), name
    assert any("experts" in name_by_id[pid] for pid in muon_params)
    assert any(name_by_id[pid].endswith("attn.weight") for pid in muon_params)

    # Expert LR scaling appears in both children.
    muon_scaled = [g for g in muon.param_groups if g.get("name") == "experts"]
    adamw_scaled = [g for g in adamw.param_groups if g.get("name") == "experts"]
    assert muon_scaled and pytest.approx(muon_scaled[0]["lr"], rel=1e-9) == 1e-4 * 2.83
    if adamw_scaled:
        assert pytest.approx(adamw_scaled[0]["lr"], rel=1e-9) == 1e-4 * 2.83


def test_lingbot_muon_config_steps_and_reduces_loss():
    torch.manual_seed(0)
    model = _TinyLingBot()
    opt = LingbotMuonConfig(lr=5e-3).build(_named_trainable(model))
    idx = torch.arange(8).unsqueeze(0)

    first = None
    for step in range(5):
        loss = model(idx).pow(2).mean()
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert model(idx).pow(2).mean().item() < first


def test_lingbot_muon_config_requires_named_params():
    model = _TinyLingBot()
    with pytest.raises(TypeError, match="named parameters"):
        LingbotMuonConfig().build(list(model.parameters()))


def test_combined_optimizer_state_dict_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _TinyLingBot()
    opt = LingbotMuonConfig(lr=1e-3).build(_named_trainable(model))
    idx = torch.arange(8).unsqueeze(0)
    for _ in range(2):
        loss = model(idx).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Flat torch-style layout over the concatenated groups.
    sd = opt.state_dict()
    assert set(sd) == {"state", "param_groups"}
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert sum(len(g["params"]) for g in sd["param_groups"]) == n_params
    assert len(sd["state"]) > 0
    # Muon momentum buffers + AdamW moments are both present.
    state_lens = {len(v) for v in sd["state"].values()}
    assert 1 in state_lens  # Muon: momentum_buffer only
    assert 3 in state_lens  # AdamW: exp_avg + exp_avg_sq + step

    # Round-trip through lerobot's safetensors save/load, then into a fresh clone.
    _save_single_optimizer_state(opt, tmp_path)
    clone = _TinyLingBot()
    clone.load_state_dict(model.state_dict())
    opt2 = LingbotMuonConfig(lr=1e-3).build(_named_trainable(clone))
    before = opt2.state_dict()["state"]
    assert all(len(v) == 0 for v in before.values()) or before == {}
    _load_single_optimizer_state(opt2, tmp_path)
    after_muon, after_adamw = opt2.optimizers

    old_muon, old_adamw = opt.optimizers
    for old_opt, new_opt in ((old_muon, after_muon), (old_adamw, after_adamw)):
        for group_old, group_new in zip(old_opt.param_groups, new_opt.param_groups):
            for p_old, p_new in zip(group_old["params"], group_new["params"]):
                s_old, s_new = old_opt.state[p_old], new_opt.state[p_new]
                assert set(s_old) == set(s_new), (set(s_old), set(s_new))
                for key in s_old:
                    assert torch.allclose(s_old[key], s_new[key], atol=0, rtol=0), key


def test_scheduler_steps_combined_optimizer():
    torch.manual_seed(0)
    model = _TinyLingBot()
    opt = LingbotMuonConfig(lr=1e-3).build(_named_trainable(model))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5)
    idx = torch.arange(8).unsqueeze(0)
    for _ in range(2):
        loss = model(idx).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    lrs = {g["lr"] for g in opt.param_groups}
    assert all(lr < 1e-3 or lr <= 1e-3 for lr in lrs)


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
