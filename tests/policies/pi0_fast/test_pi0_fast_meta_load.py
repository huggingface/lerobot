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

"""PI0FastPolicy.from_pretrained builds parameters on the meta device and matches the regular load."""

import sys

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

import transformers.utils  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0_fast import PI0FastConfig, PI0FastPolicy, modeling_pi0_fast  # noqa: E402
from lerobot.utils import import_utils  # noqa: E402
from tests.policies.common.meta_load import (  # noqa: E402
    DOWNLOAD_ARGUMENTS,
    EMBED_TOKENS,
    add_unknown_weights,
    assert_embedding_stored_first,
    assert_same,
    drop_projector_bias,
    load,
    save_checkpoint,
    store_embedding,
    use_tiny_backbone,
)


class Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture
def config(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi0_fast)
    monkeypatch.setattr(modeling_pi0_fast, "AutoProcessor", Tokenizer)
    monkeypatch.setattr(modeling_pi0_fast, "AutoTokenizer", Tokenizer)
    config = PI0FastConfig(image_resolution=(28, 28), device="cpu", dtype=torch.bfloat16)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def test_matches_regular_load(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt")
    model, used_meta = load(PI0FastPolicy, path, config, monkeypatch)
    expected, _ = load(PI0FastPolicy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)
    assert isinstance(model.action_tokenizer, Tokenizer) and isinstance(
        model.model._paligemma_tokenizer, Tokenizer
    )
    # The text head decodes action tokens; like the regular load, it gets its own copy of the word table.
    lm_head = model.model.paligemma_with_expert.paligemma.lm_head.weight
    embed_tokens = model.get_parameter(EMBED_TOKENS)
    assert torch.equal(lm_head, embed_tokens) and lm_head.data_ptr() != embed_tokens.data_ptr()


def test_skips_weights_the_policy_does_not_have(config, tmp_path, monkeypatch, capsys):
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt", edit=add_unknown_weights)
    model, used_meta = load(PI0FastPolicy, path, config, monkeypatch)
    assert "Unexpected keys when loading state dict: 2 keys" in capsys.readouterr().out
    expected, _ = load(PI0FastPolicy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_stored_embedding_next_to_lm_head_matches_regular_load(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt", edit=store_embedding)
    assert_embedding_stored_first(path)
    model, used_meta = load(PI0FastPolicy, path, config, monkeypatch)
    expected, _ = load(PI0FastPolicy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_incomplete_checkpoint_loads_the_regular_way(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt", edit=drop_projector_bias)
    model, used_meta = load(PI0FastPolicy, path, config, monkeypatch, strict=False)
    expected, _ = load(PI0FastPolicy, path, config, monkeypatch, regular=True, strict=False)
    assert not used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("complete", [True, False])
def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch, complete):
    path = save_checkpoint(
        PI0FastPolicy, config, tmp_path / "ckpt", edit=None if complete else drop_projector_bias
    )
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args)

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    _, used_meta = load(PI0FastPolicy, path, config, monkeypatch, **DOWNLOAD_ARGUMENTS)
    # One lookup, with every option, serves whichever path loads the weights.
    assert calls == [DOWNLOAD_ARGUMENTS] and used_meta == complete


def test_missing_weights_raise(config, tmp_path, monkeypatch):
    with pytest.raises(OSError, match="model.safetensors"):
        load(PI0FastPolicy, tmp_path, config, monkeypatch)


def test_missing_transformers_names_the_extra(config, tmp_path, monkeypatch):
    monkeypatch.setitem(import_utils._require_package_cache, "transformers", False)
    monkeypatch.setitem(sys.modules, "transformers.utils", None)
    with pytest.raises(ImportError, match=r"lerobot\[pi\]"):
        PI0FastPolicy.from_pretrained(tmp_path, config=config)
