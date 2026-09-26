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

"""PI05Policy.from_pretrained builds parameters on the meta device and matches the regular load."""

import pytest

pytest.importorskip("transformers")

import transformers.utils  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from tests.policies.pi0_pi05.utils.meta_load import (  # noqa: E402
    assert_same,
    load,
    save_checkpoint,
    use_tiny_backbone,
)
from tests.utils import require_cuda  # noqa: E402

RENAMES = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}


@pytest.fixture
def config(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi05)
    config = PI05Config(image_resolution=(28, 28), device="cpu", dtype="bfloat16")
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def test_matches_regular_load(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    model, used_meta = load(PI05Policy, path, config, monkeypatch)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)



def test_subclass_with_its_own_constructor(config, tmp_path, monkeypatch):
    class Subclass(PI05Policy):
        def __init__(self, config, extra=None):
            super().__init__(config)
            self.extra = extra

    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    model, used_meta = load(Subclass, path, config, monkeypatch, extra=1)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta and type(model) is Subclass and model.extra == 1
    assert_same(model, expected)

@require_cuda
def test_matches_regular_load_on_cuda(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    config.device = "cuda"
    model, used_meta = load(PI05Policy, path, config, monkeypatch)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("saved_with_memory", [True, False])
def test_proprioceptive_memory(config, tmp_path, monkeypatch, saved_with_memory):
    # A checkpoint saved without the memory projection keeps the regular path, which initializes it fresh.
    config.use_proprioceptive_memory = saved_with_memory
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    config.use_proprioceptive_memory = True
    model, used_meta = load(PI05Policy, path, config, monkeypatch)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta == saved_with_memory
    assert_same(model, expected)


@pytest.mark.parametrize("regular", [False, True])
def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch, regular):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args, **{**kwargs, "revision": None, "token": None})

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    arguments = {"revision": "v1", "token": "secret", "cache_dir": str(tmp_path / "cache")}
    load(PI05Policy, path, config, monkeypatch, regular=regular, **arguments)
    assert calls and all(call.items() >= arguments.items() for call in calls)
