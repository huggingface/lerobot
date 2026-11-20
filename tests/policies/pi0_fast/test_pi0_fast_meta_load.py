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

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

import transformers.utils  # noqa: E402
from safetensors import SafetensorError  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0_fast import PI0FastConfig, PI0FastPolicy, modeling_pi0_fast  # noqa: E402
from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS  # noqa: E402
from tests.policies.common.meta_load import (  # noqa: E402
    DOWNLOAD_ARGUMENTS,
    EMBED_TOKENS,
    assert_same,
    drop_projector_bias,
    load,
    observation,
    save_checkpoint,
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


def assert_same_loss(actual, expected):
    """Compare the training loss, since decoding actions needs the real tokenizers."""
    tokens = torch.arange(1, 7, device=expected.config.device).unsqueeze(0)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    batch = observation(expected.config) | {ACTION_TOKENS: tokens, ACTION_TOKEN_MASK: mask}
    assert torch.equal(actual.forward(batch)[0], expected.forward(batch)[0])


def test_matches_regular_load(config, tmp_path, monkeypatch):
    config.gradient_checkpointing = True
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt")
    model, used_meta = load(PI0FastPolicy, path, config, monkeypatch)
    expected, _ = load(PI0FastPolicy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert model.model.gradient_checkpointing_enabled is expected.model.gradient_checkpointing_enabled
    assert model.model.gradient_checkpointing_enabled is True
    assert_same(model, expected)
    assert_same_loss(model, expected)
    assert isinstance(model.action_tokenizer, Tokenizer) and isinstance(
        model.model._paligemma_tokenizer, Tokenizer
    )
    # The text head decodes action tokens; like the regular load, it gets its own copy of the word table.
    lm_head = model.model.paligemma_with_expert.paligemma.lm_head.weight
    embed_tokens = model.get_parameter(EMBED_TOKENS)
    assert torch.equal(lm_head, embed_tokens) and lm_head.data_ptr() != embed_tokens.data_ptr()


def test_subclass_conversion_runs_the_regular_way(config, tmp_path, monkeypatch):
    class Subclass(PI0FastPolicy):
        def _convert_openpi_state_dict(self, state_dict):
            self.converted = True
            return super()._convert_openpi_state_dict(state_dict)

    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt")
    model, used_meta = load(Subclass, path, config, monkeypatch)
    assert not used_meta and model.converted


def test_strict_refuses_a_checkpoint_that_leaves_a_weight_unfilled(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt", edit=drop_projector_bias)
    with pytest.raises(RuntimeError, match="multi_modal_projector.linear.bias"):
        load(PI0FastPolicy, path, config, monkeypatch, strict=True)


def test_strict_refuses_an_unreadable_checkpoint(config, tmp_path, monkeypatch):
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"not a safetensors file")
    with pytest.raises(SafetensorError):
        load(PI0FastPolicy, path, config, monkeypatch, strict=True)


def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt", edit=drop_projector_bias)
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args)

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    _, used_meta = load(PI0FastPolicy, path, config, monkeypatch, **DOWNLOAD_ARGUMENTS)
    # One lookup, with every option, serves whichever path loads the weights.
    assert calls == [DOWNLOAD_ARGUMENTS] and not used_meta


@pytest.mark.parametrize("regular", [False, True], ids=["meta", "regular"])
def test_missing_weights_raise(config, tmp_path, monkeypatch, regular):
    with pytest.raises(OSError, match="model.safetensors"):
        load(PI0FastPolicy, tmp_path, config, monkeypatch, regular=regular)


def test_loading_inside_torch_device_meta_raises(config, tmp_path):
    # Nothing would place the weights, so it must fail instead of returning a policy without them.
    path = save_checkpoint(PI0FastPolicy, config, tmp_path / "ckpt")
    with pytest.raises(NotImplementedError, match="meta tensor"), torch.device("meta"):
        PI0FastPolicy.from_pretrained(path, config=config)
