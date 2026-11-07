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

import logging

import pytest
import torch

pytest.importorskip("transformers")

import transformers.utils  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from tests.policies.common.meta_load import (  # noqa: E402
    DOWNLOAD_ARGUMENTS,
    LOADER_LOGGER,
    add_unknown_weights,
    assert_same,
    assert_same_actions,
    drop_projector_bias,
    load,
    save_checkpoint,
    use_tiny_backbone,
)

RENAMES = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}


@pytest.fixture
def config(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi05)
    config = PI05Config(image_resolution=(28, 28), device="cpu", dtype=torch.bfloat16)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def test_matches_regular_load(config, tmp_path, monkeypatch, caplog):
    config.gradient_checkpointing = True
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES, add_unknown_weights)
    with caplog.at_level(logging.WARNING):
        model, used_meta = load(PI05Policy, path, config, monkeypatch)
    unexpected = ["model.state_mlp_in.bias", "model.state_mlp_in.weight"]
    assert f"Unexpected key(s) when loading model: {unexpected}" in caplog.text
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert model.model.gradient_checkpointing_enabled is expected.model.gradient_checkpointing_enabled
    assert model.model.gradient_checkpointing_enabled is True
    assert_same(model, expected)
    assert_same_actions(model, expected)


def test_proprioceptive_memory(config, tmp_path, monkeypatch, caplog):
    # A checkpoint saved without the memory projection keeps the regular path, which initializes it fresh.
    # That is the expected way to start from a base checkpoint, so it is not reported as a fallback.
    config.use_proprioceptive_memory = False
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    config.use_proprioceptive_memory = True
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(PI05Policy, path, config, monkeypatch)
    assert "regular way" not in caplog.text
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES, drop_projector_bias)
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args)

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    _, used_meta = load(PI05Policy, path, config, monkeypatch, **DOWNLOAD_ARGUMENTS)
    # One lookup, with every option, serves whichever path loads the weights.
    assert calls == [DOWNLOAD_ARGUMENTS] and not used_meta


@pytest.mark.parametrize("regular", [False, True], ids=["meta", "regular"])
def test_missing_weights_raise(config, tmp_path, monkeypatch, regular):
    with pytest.raises(OSError, match="model.safetensors"):
        load(PI05Policy, tmp_path, config, monkeypatch, regular=regular)


def test_loading_inside_torch_device_meta_raises(config, tmp_path):
    # Nothing would place the weights, so it must fail instead of returning a policy without them.
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    with pytest.raises(NotImplementedError, match="meta tensor"), torch.device("meta"):
        PI05Policy.from_pretrained(path, config=config)
