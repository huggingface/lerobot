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

"""Helpers for the tests that load pi0, pi05 and pi0_fast with their parameters on the meta device."""

import sys

import torch
from safetensors.torch import save_file
from transformers import PaliGemmaConfig

from lerobot.policies.common import openpi_checkpoint

EMBED_TOKENS = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"


def use_tiny_backbone(monkeypatch, modeling_module):
    """Make the policy in `modeling_module` have a few million parameters instead of billions."""
    tiny = modeling_module.GemmaConfig(
        width=16, depth=1, mlp_dim=32, num_heads=1, num_kv_heads=1, head_dim=16
    )
    monkeypatch.setattr(modeling_module, "get_gemma_config", lambda variant: tiny)
    vision = {
        "model_type": "siglip_vision_model",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "vision_use_head": False,
    }
    configs = {
        "paligemma": lambda: PaliGemmaConfig(vision_config=vision),
        "gemma": modeling_module.CONFIG_MAPPING["gemma"],
    }
    monkeypatch.setattr(modeling_module, "CONFIG_MAPPING", configs)


def save_checkpoint(policy_cls, config, path, renames=None, edit=None):
    """Save a checkpoint in the OpenPI layout that `_fix_pytorch_state_dict_keys` converts."""
    torch.manual_seed(0)
    state_dict = {
        key.removeprefix("model."): tensor.float() for key, tensor in policy_cls(config).state_dict().items()
    }
    del state_dict[EMBED_TOKENS.removeprefix("model.")]
    for old, new in (renames or {}).items():
        state_dict[new] = state_dict.pop(old)
    if edit is not None:
        edit(state_dict)
    path.mkdir()
    save_file(state_dict, path / "model.safetensors")
    return path


def load(policy_cls, path, config, monkeypatch, regular=False, **kwargs):
    """Load `path` and return the policy and whether it is the one built with parameters on meta."""
    built_on_meta = []
    load_into_meta = openpi_checkpoint._load_state_dict_into_meta_model

    def check_meta(model, *args):
        if all(param.is_meta for param in model.parameters()):
            built_on_meta.append(model)
        load_into_meta(model, *args)

    with monkeypatch.context() as patch:
        patch.setattr(openpi_checkpoint, "_load_state_dict_into_meta_model", check_meta)
        if regular:
            # `from_pretrained` looks the fast path up in the module that defines it.
            module = sys.modules[policy_cls.from_pretrained.__module__]
            patch.setattr(module, "load_complete_checkpoint", lambda *args, **kwargs: None)
        torch.manual_seed(1)
        model = policy_cls.from_pretrained(path, config=config, **kwargs)
    return model, len(built_on_meta) == 1 and built_on_meta[0] is model


def assert_same(actual, expected):
    def tensors(model):
        return {**dict(model.named_parameters()), **dict(model.named_buffers())}

    actual_tensors, expected_tensors = tensors(actual), tensors(expected)
    assert actual_tensors.keys() == expected_tensors.keys()
    for name, tensor in expected_tensors.items():
        other = actual_tensors[name]
        assert not other.is_meta, name
        assert other.dtype == tensor.dtype, name
        assert other.device == tensor.device, name
        assert other.requires_grad == tensor.requires_grad, name
        assert torch.equal(other, tensor), name
    assert actual.training == expected.training
