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

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0 import PI0Config, PI0Policy, modeling_pi0  # noqa: E402
from lerobot.policies.pretrained import _parameters_on_meta  # noqa: E402
from tests.policies.pi0_pi05.utils.meta_load import (  # noqa: E402
    EMBED_TOKENS,
    assert_same,
    load,
    save_checkpoint,
    use_tiny_backbone,
)
from tests.utils import require_cuda  # noqa: E402

RENAMES = {f"action_time_mlp_in.{name}": f"time_mlp_in.{name}" for name in ("weight", "bias")}


@pytest.fixture
def config(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi0)
    config = PI0Config(image_resolution=(28, 28), device="cpu", dtype="bfloat16")
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def save(config, path, edit=None):
    return save_checkpoint(PI0Policy, config, path, RENAMES, edit)


def test_matches_regular_load(config, tmp_path, monkeypatch):
    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)
    lm_head = model.model.paligemma_with_expert.paligemma.lm_head.weight
    embed_tokens = model.get_parameter(EMBED_TOKENS)
    assert torch.equal(lm_head, embed_tokens) and lm_head.data_ptr() != embed_tokens.data_ptr()


@require_cuda
def test_matches_regular_load_on_cuda(config, tmp_path, monkeypatch):
    path = save(config, tmp_path / "ckpt")
    config.device = "cuda"
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_frozen_parameters_keep_requires_grad(config, tmp_path, monkeypatch):
    config.freeze_vision_encoder = True
    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_subclass_with_its_own_constructor(config, tmp_path, monkeypatch):
    class Subclass(PI0Policy):
        def __init__(self, config, extra=None):
            super().__init__(config)
            self.extra = extra

    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(Subclass, path, config, monkeypatch, extra=1)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta and type(model) is Subclass and model.extra == 1
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
    path = save(config, tmp_path / "ckpt", edit)
    model, used_meta = load(PI0Policy, path, config, monkeypatch, strict=strict)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True, strict=strict)
    assert not used_meta
    assert_same(model, expected)


def test_bigger_action_space_loads_the_regular_way(config, tmp_path, monkeypatch):
    path = save(config, tmp_path / "ckpt")
    config.max_action_dim = 48
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


def test_unreadable_checkpoint_loads_the_regular_way(config, tmp_path, monkeypatch):
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"not a safetensors file")
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("regular", [False, True])
def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch, regular):
    path = save(config, tmp_path / "ckpt")
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args, **{**kwargs, "revision": None, "token": None})

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    arguments = {"revision": "v1", "token": "secret", "cache_dir": str(tmp_path / "cache")}
    load(PI0Policy, path, config, monkeypatch, regular=regular, **arguments)
    assert calls and all(call.items() >= arguments.items() for call in calls)
