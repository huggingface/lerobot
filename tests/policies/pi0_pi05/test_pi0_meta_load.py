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

"""PI0Policy.from_pretrained builds parameters on the meta device and matches the regular load."""

import pytest
import torch

pytest.importorskip("transformers")

import transformers.utils  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import PaliGemmaConfig  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0 import PI0Config, PI0Policy, modeling_pi0  # noqa: E402
from lerobot.policies.pretrained import _parameters_on_meta  # noqa: E402
from tests.utils import require_cuda  # noqa: E402

EMBED_TOKENS = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"


@pytest.fixture
def config(monkeypatch):
    """A PI0 config whose model has a few million parameters instead of 3.5 billion."""
    tiny = modeling_pi0.GemmaConfig(width=16, depth=1, mlp_dim=32, num_heads=1, num_kv_heads=1, head_dim=16)
    monkeypatch.setattr(modeling_pi0, "get_gemma_config", lambda variant: tiny)
    vision = {
        "model_type": "siglip_vision_model",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "vision_use_head": False,
    }
    configs = {
        "paligemma": lambda: PaliGemmaConfig(vision_config=vision),
        "gemma": modeling_pi0.CONFIG_MAPPING["gemma"],
    }
    monkeypatch.setattr(modeling_pi0, "CONFIG_MAPPING", configs)
    config = PI0Config(image_resolution=(28, 28), device="cpu", dtype="bfloat16")
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def save_checkpoint(config, path, edit=None):
    """Save a checkpoint in the OpenPI layout that `_fix_pytorch_state_dict_keys` converts."""
    torch.manual_seed(0)
    state_dict = {
        key.removeprefix("model."): tensor.float() for key, tensor in PI0Policy(config).state_dict().items()
    }
    del state_dict[EMBED_TOKENS.removeprefix("model.")]
    for name in ("weight", "bias"):
        state_dict[f"time_mlp_in.{name}"] = state_dict.pop(f"action_time_mlp_in.{name}")
    if edit is not None:
        edit(state_dict)
    path.mkdir()
    save_file(state_dict, path / "model.safetensors")
    return path


def load(path, config, monkeypatch, regular=False, policy_cls=PI0Policy, **kwargs):
    """Load `path` and return the policy and whether it is the one built with parameters on meta."""
    built_on_meta = []
    load_into_meta = modeling_pi0._load_state_dict_into_meta_model

    def check_meta(model, *args):
        if all(param.is_meta for param in model.parameters()):
            built_on_meta.append(model)
        load_into_meta(model, *args)

    with monkeypatch.context() as patch:
        patch.setattr(modeling_pi0, "_load_state_dict_into_meta_model", check_meta)
        if regular:
            patch.setattr(PI0Policy, "_from_complete_checkpoint", classmethod(lambda cls, *a, **k: None))
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


def test_matches_regular_load(config, tmp_path, monkeypatch):
    path = save_checkpoint(config, tmp_path / "ckpt")
    model, used_meta = load(path, config, monkeypatch)
    expected, _ = load(path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)
    lm_head = model.model.paligemma_with_expert.paligemma.lm_head.weight
    embed_tokens = model.get_parameter(EMBED_TOKENS)
    assert torch.equal(lm_head, embed_tokens) and lm_head.data_ptr() != embed_tokens.data_ptr()


@require_cuda
def test_matches_regular_load_on_cuda(config, tmp_path, monkeypatch):
    path = save_checkpoint(config, tmp_path / "ckpt")
    config.device = "cuda"
    model, used_meta = load(path, config, monkeypatch)
    expected, _ = load(path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_frozen_parameters_keep_requires_grad(config, tmp_path, monkeypatch):
    config.freeze_vision_encoder = True
    path = save_checkpoint(config, tmp_path / "ckpt")
    model, used_meta = load(path, config, monkeypatch)
    expected, _ = load(path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_subclass_with_its_own_constructor(config, tmp_path, monkeypatch):
    class Subclass(PI0Policy):
        def __init__(self, config):
            super().__init__(config)

    path = save_checkpoint(config, tmp_path / "ckpt")
    model, used_meta = load(path, config, monkeypatch, policy_cls=Subclass)
    expected, _ = load(path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_meta_build_draws_no_random_numbers(config):
    # Falling back builds the model again, so it must see the same random state as without the fast path.
    state = torch.random.get_rng_state()
    with _parameters_on_meta():
        PI0Policy(config)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "edit",
    [
        lambda sd: sd.pop("state_proj.bias"),
        lambda sd: sd.update(extra=torch.ones(1)),
        lambda sd: sd.update({"state_proj.bias": torch.ones(3)}),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_other_checkpoints_load_the_regular_way(config, tmp_path, monkeypatch, edit, strict):
    path = save_checkpoint(config, tmp_path / "ckpt", edit)
    model, used_meta = load(path, config, monkeypatch, strict=strict)
    expected, _ = load(path, config, monkeypatch, regular=True, strict=strict)
    assert not used_meta
    assert_same(model, expected)


def test_bigger_action_space_loads_the_regular_way(config, tmp_path, monkeypatch):
    path = save_checkpoint(config, tmp_path / "ckpt")
    config.max_action_dim = 48
    model, used_meta = load(path, config, monkeypatch)
    expected, _ = load(path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


def test_unreadable_checkpoint_loads_the_regular_way(config, tmp_path, monkeypatch):
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"not a safetensors file")
    model, used_meta = load(path, config, monkeypatch)
    expected, _ = load(path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("regular", [False, True])
def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch, regular):
    path = save_checkpoint(config, tmp_path / "ckpt")
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args, **{**kwargs, "revision": None, "token": None})

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    arguments = {"revision": "v1", "token": "secret", "cache_dir": str(tmp_path / "cache")}
    load(path, config, monkeypatch, regular=regular, **arguments)
    assert calls and all(call.items() >= arguments.items() for call in calls)
