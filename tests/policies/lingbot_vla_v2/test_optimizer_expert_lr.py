# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations and
# under the License is distributed.

"""CPU test that lingbot_adamw reproduces the upstream MoE expert-LR param grouping
(upstream train_lingbotvla.py::get_moe_param_groups: routed experts at
lr * (num_experts / top_k) ** 0.5, everything else — including the ViT, which
upstream trains at the base LR — at the base LR)."""

import pytest

torch = pytest.importorskip("torch")

from lerobot.optim.optimizers import LingbotAdamWConfig  # noqa: E402


def _named_params():
    """Name -> Parameter mapping shaped like the vendored dual-stream model."""
    return {
        "qwenvl.model.visual.blocks.0.mlp.down_proj.weight": torch.nn.Parameter(torch.randn(4, 4)),
        "qwenvl.model.language.layers.3.self_attn.q_proj.weight": torch.nn.Parameter(torch.randn(4, 4)),
        "qwen_expert.model.layers.0.mlp.experts.down_weight": torch.nn.Parameter(torch.randn(8, 4)),
        "qwen_expert.model.layers.35.mlp.experts.up_weight": torch.nn.Parameter(torch.randn(4, 8)),
        "qwen_expert.model.layers.35.mlp.shared_expert.gate_proj.weight": torch.nn.Parameter(torch.randn(4, 4)),
        "qwen_expert.model.layers.35.mlp.gate.weight": torch.nn.Parameter(torch.randn(4, 32)),
        "action_in_proj.weight": torch.nn.Parameter(torch.randn(4, 4)),
    }


def test_expert_group_gets_scaled_lr():
    named = _named_params()
    cfg = LingbotAdamWConfig(lr=1e-4, expert_lr_scale=(32 / 4) ** 0.5)
    opt = cfg.build(dict(named))
    groups = {g["name"]: g for g in opt.param_groups}
    assert set(groups) == {"experts", "other"}

    experts = {id(p) for p in groups["experts"]["params"]}
    expected_experts = {id(p) for name, p in named.items() if ".mlp.experts." in name}
    assert experts == expected_experts  # gate / shared_expert / ViT stay in "other"
    assert groups["experts"]["lr"] == pytest.approx(1e-4 * (32 / 4) ** 0.5)
    assert groups["other"]["lr"] == pytest.approx(1e-4)
    assert opt.defaults["weight_decay"] == 0.0
    assert opt.defaults["betas"] == (0.9, 0.95)


def test_scale_one_matches_plain_adamw_grouping():
    cfg = LingbotAdamWConfig(lr=1e-5, expert_lr_scale=1.0)
    opt = cfg.build(_named_params())
    lrs = {g["lr"] for g in opt.param_groups}
    assert lrs == {1e-5}
    total = sum(len(g["params"]) for g in opt.param_groups)
    assert total == len(_named_params())


def test_frozen_params_excluded_and_iterable_fallback():
    params = _named_params()
    frozen_key = "action_in_proj.weight"
    params[frozen_key].requires_grad_(False)
    opt = LingbotAdamWConfig(lr=1e-4, expert_lr_scale=2.83).build(params)
    n_optimized = sum(len(g["params"]) for g in opt.param_groups)
    assert n_optimized == len(params) - 1

    # Plain iterable (no names) falls back to a single base-LR group.
    flat = list(_named_params().values())
    opt_flat = LingbotAdamWConfig(lr=1e-4, expert_lr_scale=2.83).build(flat)
    assert len(opt_flat.param_groups) == 1
    assert opt_flat.param_groups[0]["lr"] == 1e-4


def test_policy_preset_wires_the_scale():
    pytest.importorskip("lerobot.policies.lingbot_vla_v2")
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

    cfg = LingbotVLAV2Config()
    cfg.use_moe = True
    cfg.token_num_experts = 32
    cfg.token_top_k = 4
    cfg.use_moe_expert_lr = True
    preset = cfg.get_optimizer_preset()
    assert preset.expert_lr_scale == pytest.approx((32 / 4) ** 0.5)

    cfg.use_moe_expert_lr = False
    assert cfg.get_optimizer_preset().expert_lr_scale == 1.0
