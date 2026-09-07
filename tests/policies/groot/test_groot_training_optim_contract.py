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

"""Isaac-GR00T N1.7 optimizer/scheduler/precision training contract.

Pins the LeRobot GR00T fine-tuning recipe to the native Isaac-GR00T contract:
AdamW(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, grad clip 1.0),
HF cosine schedule with ~5% warmup over the actual update count, FP32 master
parameters under BF16 autocast, transformers-style weight-decay grouping, the
frozen LM-head weight tie, and episode-tail exclusion for incomplete chunks.
"""

import pytest
import torch

from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy


def test_groot_n1_7_optimizer_matches_isaac_training_contract():
    optimizer = GrootConfig().get_optimizer_preset()

    assert optimizer.lr == pytest.approx(1e-4)
    assert optimizer.betas == pytest.approx((0.9, 0.999))
    assert optimizer.eps == pytest.approx(1e-8)
    assert optimizer.weight_decay == pytest.approx(1e-5)
    assert optimizer.grad_clip_norm == pytest.approx(1.0)


def test_groot_n1_7_sampler_excludes_incomplete_action_tails():
    config = GrootConfig(chunk_size=16, n_action_steps=16)

    assert len(config.action_delta_indices) == 16
    assert config.drop_n_last_frames == 15


def test_groot_n1_7_scheduler_matches_isaac_hf_cosine_contract():
    pytest.importorskip("diffusers", reason="the scheduler preset requires the `groot` extra (diffusers)")
    config = GrootConfig(max_steps=20_000)
    scheduler_config = config.get_scheduler_preset()

    assert isinstance(scheduler_config, DiffuserSchedulerConfig)
    assert scheduler_config.name == "cosine"
    assert scheduler_config.num_warmup_steps == 1_000

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=config.optimizer_lr)
    scheduler = scheduler_config.build(optimizer, num_training_steps=20_000)
    lr_factor = scheduler.lr_lambdas[0]

    assert lr_factor(0) == pytest.approx(0.0)
    assert lr_factor(1_000) == pytest.approx(1.0)
    assert lr_factor(10_500) == pytest.approx(0.5)
    assert lr_factor(20_000) == pytest.approx(0.0, abs=1e-12)


def test_groot_n1_7_scheduler_rounds_fractional_warmup_up_like_transformers():
    scheduler_config = GrootConfig(max_steps=777).get_scheduler_preset()

    assert scheduler_config.num_warmup_steps == 39


def test_groot_n1_7_model_parameters_use_fp32_checkpoint_and_optimizer_precision():
    module = torch.nn.Module()
    module.trainable = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    module.frozen = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False)

    GrootPolicy._cast_model_parameters_to_fp32(module)

    assert module.trainable.dtype == torch.float32
    assert module.frozen.dtype == torch.float32


def test_groot_n1_7_ties_unused_qwen_lm_head_to_frozen_input_embeddings():
    class DummyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(7, 3)
            self.lm_head = torch.nn.Linear(3, 7, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    model = DummyQwen()
    _tie_unused_qwen_lm_head(model)

    assert model.lm_head.weight is model.embed_tokens.weight
    assert len(list(model.parameters())) == 1


def test_groot_n1_7_optimizer_groups_match_transformers_weight_decay_rules():
    pytest.importorskip(
        "transformers", reason="weight-decay grouping requires the `groot` extra (transformers)"
    )
    module = torch.nn.Module()
    module.linear = torch.nn.Linear(3, 2)
    module.norm = torch.nn.LayerNorm(2)
    module.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    groups = GrootPolicy._build_weight_decay_parameter_groups(module)

    assert len(groups) == 2
    assert "weight_decay" not in groups[0]
    assert groups[1]["weight_decay"] == 0.0
    assert groups[0]["params"] == [module.linear.weight]
    assert {id(parameter) for parameter in groups[1]["params"]} == {
        id(module.linear.bias),
        id(module.norm.weight),
        id(module.norm.bias),
    }
