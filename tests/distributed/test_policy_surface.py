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
"""The declarative policy surface and its distributed-side consumers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.accelerator import FSDPConfig
from lerobot.distributed import set_fsdp_wrap_modules, strip_accelerate_cp_hooks
from lerobot.policies.pretrained import PreTrainedPolicy


class TestDeclarativeAttributes:
    def test_base_defaults(self):
        assert PreTrainedPolicy._fsdp_wrap_modules is None
        assert PreTrainedPolicy._fsdp_forward_methods == ("select_action", "predict_action_chunk")
        assert PreTrainedPolicy.supports_gradient_checkpointing is False
        assert PreTrainedPolicy._cp_plan is None

    def test_act_wrap_units_name_real_classes(self):
        """The declared class names must track the modeling code — this test pins the drift."""
        from lerobot.policies.act import modeling_act

        for name in modeling_act.ACTPolicy._fsdp_wrap_modules:
            assert isinstance(getattr(modeling_act, name), type), name

    def test_fastwam_wrap_units_name_real_classes(self):
        from lerobot.policies.fastwam import modeling_fastwam
        from lerobot.policies.fastwam.wan import modular

        for name in modeling_fastwam.FastWAMPolicy._fsdp_wrap_modules:
            assert isinstance(getattr(modular, name), type), name


class _SelfAttn(nn.Module):
    def forward(self, x, attention_mask=None, is_causal=False):
        return x, attention_mask, is_causal


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SelfAttn()


class TestStripAccelerateCpHooks:
    def test_strips_the_real_accelerate_hook_and_restores_mask_semantics(self):
        """Attach accelerate's actual mask-stripping hook, strip it, verify masks survive."""
        from accelerate.big_modeling import _attach_context_parallel_hooks

        model = _TinyModel()
        mask = torch.ones(2, 2)

        _attach_context_parallel_hooks(model)
        _, hooked_mask, hooked_causal = model.self_attn(torch.zeros(1), attention_mask=mask)
        assert hooked_mask is None and hooked_causal is True  # the hazard is real

        assert strip_accelerate_cp_hooks(model) == 1
        _, clean_mask, clean_causal = model.self_attn(torch.zeros(1), attention_mask=mask)
        assert clean_mask is mask and clean_causal is False
        assert not model.self_attn._forward_pre_hooks
        assert not model.self_attn._forward_pre_hooks_with_kwargs

    def test_user_hooks_survive(self):
        model = _TinyModel()
        model.self_attn.register_forward_pre_hook(lambda m, args: None)
        assert strip_accelerate_cp_hooks(model) == 0
        assert len(model.self_attn._forward_pre_hooks) == 1


class _DeclaredPolicy:
    _fsdp_wrap_modules = ["DeclaredBlock"]


class _UndeclaredPolicy:
    _fsdp_wrap_modules = None


def _accelerator_with(plugin) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))


class TestSetFsdpWrapModules:
    def test_policy_declaration_fills_plugin(self):
        plugin = FSDPConfig().build_plugin()
        set_fsdp_wrap_modules(_accelerator_with(plugin), _DeclaredPolicy())
        assert plugin.transformer_cls_names_to_wrap == ["DeclaredBlock"]

    def test_user_override_wins(self):
        plugin = FSDPConfig(wrap_modules=["UserBlock"]).build_plugin()
        set_fsdp_wrap_modules(_accelerator_with(plugin), _DeclaredPolicy())
        assert plugin.transformer_cls_names_to_wrap == ["UserBlock"]

    def test_no_wrap_source_fails_loudly(self):
        plugin = FSDPConfig().build_plugin()
        with pytest.raises(ValueError, match="_fsdp_wrap_modules"):
            set_fsdp_wrap_modules(_accelerator_with(plugin), _UndeclaredPolicy())

    def test_size_based_policy_needs_no_names(self):
        plugin = FSDPConfig(min_num_params=1024).build_plugin()
        set_fsdp_wrap_modules(_accelerator_with(plugin), _UndeclaredPolicy())
        assert plugin.transformer_cls_names_to_wrap is None

    def test_non_sharded_run_is_noop(self):
        set_fsdp_wrap_modules(_accelerator_with(None), _UndeclaredPolicy())
