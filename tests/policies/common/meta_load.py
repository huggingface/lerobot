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
from safetensors.torch import load_file, save_file
from transformers import PaliGemmaConfig

from lerobot.policies.common import openpi_checkpoint
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

EMBED_TOKENS = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
LOADER_LOGGER = openpi_checkpoint.__name__

# Every option the weights lookup gets, none at its default value. As in `PreTrainedPolicy.from_pretrained`,
# `resume_download` and `proxies` only go to the config lookup, since the Hub ignores them.
DOWNLOAD_ARGUMENTS = {
    "force_download": True,
    "token": "secret",
    "cache_dir": "/cache",
    "local_files_only": True,
    "revision": "v1",
}


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
    paligemma = modeling_module.PaliGemmaForConditionalGenerationWithPiGemma

    def tiny_paligemma(config):
        # The policies project images to Gemma 2B's width; the tiny text model needs its own width to run.
        config.vision_config.projection_dim = config.text_config.hidden_size
        return paligemma(config=config)

    monkeypatch.setattr(modeling_module, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_paligemma)


def drop_projector_bias(state_dict):
    """Make a checkpoint that the meta-device path turns down."""
    del state_dict["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"]


def add_unknown_weights(state_dict):
    """Add weights the policy does not have, as a checkpoint trained by a variant with an extra layer holds."""
    state_dict["state_mlp_in.weight"] = torch.ones(4, 8)
    state_dict["state_mlp_in.bias"] = torch.ones(4)


def use_legacy_vision_names(state_dict):
    """Name the SigLIP weights as checkpoints saved before transformers 5.6 do, like the published ones."""
    for key in [key for key in state_dict if ".vision_tower." in key]:
        state_dict[key.replace(".vision_tower.", ".vision_tower.vision_model.")] = state_dict.pop(key)


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
    built_on_meta, read = [], {}
    load_into_meta = openpi_checkpoint._load_state_dict_into_meta_model

    def read_file(filename):
        tensors = load_file(filename)
        read.update((tensor.data_ptr(), tensor.clone()) for tensor in tensors.values())
        return tensors

    def check_meta(model, state_dict, device):
        if all(param.is_meta for param in model.parameters()):
            built_on_meta.append(model)
        # Each tensor moves to the policy's device as it is copied out of the file.
        assert device == config.device
        # The conversions only rename, so each tensor is one read from the file, its values unchanged.
        for name, tensor in state_dict.items():
            original = read.get(tensor.data_ptr())
            assert original is not None and torch.equal(tensor, original), name
        load_into_meta(model, state_dict, device)

    with monkeypatch.context() as patch:
        patch.setattr(openpi_checkpoint, "load_file", read_file)
        patch.setattr(openpi_checkpoint, "_load_state_dict_into_meta_model", check_meta)
        if regular:
            # `from_pretrained` looks the fast path up in the module that defines it.
            module = sys.modules[policy_cls.from_pretrained.__module__]
            patch.setattr(module, "load_complete_checkpoint", lambda *args, **kwargs: None)
        torch.manual_seed(1)
        model = policy_cls.from_pretrained(path, config=config, **kwargs)
    return model, len(built_on_meta) == 1 and built_on_meta[0] is model


def observation(config):
    """Return one observation for the tiny policies, on the policy's device."""
    generator = torch.Generator().manual_seed(2)
    batch = {
        "observation.images.base_0_rgb": torch.rand(1, 3, 28, 28, generator=generator),
        "observation.state": torch.randn(1, 8, generator=generator),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 100, (1, 8), generator=generator),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 8, dtype=torch.bool),
    }
    return {key: value.to(config.device) for key, value in batch.items()}


def assert_same_actions(actual, expected):
    """Compare predicted actions, which also depend on any state a constructor keeps outside its modules."""
    config = expected.config
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, generator=generator).to(config.device)
    batch = observation(config)
    actions = actual.predict_action_chunk(batch, noise=noise)
    assert torch.equal(actions, expected.predict_action_chunk(batch, noise=noise))


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
