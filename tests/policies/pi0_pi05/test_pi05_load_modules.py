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

"""load_checkpoint_modules loads only some submodules of a PI05Policy, with the same weights as a full load."""

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.openpi_checkpoint import load_checkpoint_modules  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from tests.policies.common.meta_load import (  # noqa: E402
    EMBED_TOKENS,
    load,
    save_checkpoint,
    use_legacy_vision_names,
    use_tiny_backbone,
)

RENAMES = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}
EXPERT = "model.paligemma_with_expert.gemma_expert.model"
VISION = "model.paligemma_with_expert.paligemma.model.vision_tower"


@pytest.fixture(params=[torch.bfloat16, torch.float32])
def config(monkeypatch, request):
    use_tiny_backbone(monkeypatch, modeling_pi05)
    config = PI05Config(image_resolution=(28, 28), device="cpu", dtype=request.param)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def test_loads_only_the_listed_modules(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    expected, _ = load(PI05Policy, path, config, monkeypatch)
    modules = [EXPERT, "model.time_mlp_in", EMBED_TOKENS.removesuffix(".weight")]

    part = load_checkpoint_modules(PI05Policy, str(path / "model.safetensors"), config, modules)

    loaded = dict(part.named_parameters())
    reference = dict(expected.named_parameters())
    assert loaded and all(name.startswith(tuple(f"{m}." for m in modules)) for name in loaded)
    for name, tensor in loaded.items():
        assert tensor.dtype == reference[name].dtype, name
        assert torch.equal(tensor, reference[name]), name
    # The renamed time MLP and the embeddings copied from lm_head load as in a full load.
    assert "model.time_mlp_in.weight" in loaded and EMBED_TOKENS in loaded
    # Modules left out are gone rather than holding meta tensors, so an export holds only the part.
    assert part.model.paligemma_with_expert.paligemma.model.vision_tower is None
    assert part.model.action_out_proj is None
    assert not any(t.is_meta for t in (*part.parameters(), *part.buffers()))
    assert not part.training
    # An export loads no modules to read the policy's settings.
    empty = load_checkpoint_modules(PI05Policy, str(path / "model.safetensors"), config, [])
    assert list(empty.parameters()) == [] and empty.config.chunk_size == config.chunk_size


def test_loads_a_vision_tower_saved_before_transformers_flattened_it(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES, use_legacy_vision_names)
    saved = load_file(path / "model.safetensors")
    assert any(".vision_tower.vision_model." in key for key in saved)
    # The full load on meta-device parameters renames them too, rather than falling back to the regular load.
    expected, used_meta = load(PI05Policy, path, config, monkeypatch)
    assert used_meta

    part = load_checkpoint_modules(PI05Policy, str(path / "model.safetensors"), config, [VISION])

    reference = dict(expected.named_parameters())
    loaded = dict(part.named_parameters())
    assert loaded
    for name, tensor in loaded.items():
        assert torch.equal(tensor, reference[name]), name
        file_key = name.removeprefix("model.").replace(".vision_tower.", ".vision_tower.vision_model.")
        assert torch.equal(tensor, saved[file_key].to(tensor.dtype)), name


def test_raises_when_the_file_misses_a_listed_weight(config, tmp_path):
    def drop_vision_weight(state_dict):
        del state_dict[next(k for k in state_dict if k.startswith(VISION.removeprefix("model.")))]

    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES, drop_vision_weight)

    with pytest.raises(RuntimeError, match="does not hold"):
        load_checkpoint_modules(PI05Policy, str(path / "model.safetensors"), config, [VISION])


def test_refuses_a_class_that_did_not_opt_in(config, tmp_path):
    class Subclass(PI05Policy):
        pass

    with pytest.raises(ValueError, match="meta device"):
        load_checkpoint_modules(Subclass, str(tmp_path / "model.safetensors"), config, [EXPERT])


def test_a_policy_loaded_in_parts_switches_to_evaluation_mode(config, tmp_path):
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    part = load_checkpoint_modules(PI05Policy, str(path / "model.safetensors"), config, [EXPERT])

    part.eval()

    assert not any(module.training for module in part.modules())
