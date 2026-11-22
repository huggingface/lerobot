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

import json
import logging

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.checkpoint import SafetensorsCheckpoint, safetensors_shard_names
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import rename_legacy_vision_model_keys


def write_shards(tmp_path, *states):
    files = []
    for i, state in enumerate(states):
        file = tmp_path / f"model-{i}.safetensors"
        save_file(state, file)
        files.append(file)
    return files


def test_strict_load_rejects_missing_empty_buffer(tmp_path):
    model = nn.Module()
    model.register_buffer("a", torch.empty(0))
    model.register_buffer("b", torch.empty(0))
    model.register_buffer("c", torch.ones(1))
    files = write_shards(tmp_path, {"a": torch.empty(0)}, {"c": torch.zeros(1)})
    with pytest.raises(RuntimeError, match="missing keys.*b"):
        PreTrainedPolicy._load_weight_files(model, files, "cpu", True)


@pytest.mark.parametrize("alias", ["a", "b"])
def test_strict_load_accepts_tied_weights(tmp_path, alias):
    model = nn.Module()
    model.register_parameter("a", nn.Parameter(torch.zeros(3)))
    model.register_parameter("b", model.a)
    model.register_buffer("c", torch.zeros(1))
    files = write_shards(tmp_path, {alias: torch.ones(3)}, {"c": torch.ones(1)})
    PreTrainedPolicy._load_weight_files(model, files, "cpu", True)
    assert torch.equal(model.a, torch.ones(3))
    assert model.a is model.b


def test_strict_load_uses_renamed_keys_and_keeps_warnings(tmp_path, caplog):
    model = nn.Module()
    model.vision_tower = nn.Linear(2, 2)
    model.register_load_state_dict_pre_hook(rename_legacy_vision_model_keys)

    def warn(*args):
        logging.warning("load hook warning")

    model.register_load_state_dict_pre_hook(warn)
    files = write_shards(
        tmp_path,
        {"vision_tower.vision_model.weight": torch.ones(2, 2)},
        {"vision_tower.vision_model.bias": torch.ones(2)},
    )
    PreTrainedPolicy._load_weight_files(model, files, "cpu", True)
    assert torch.equal(model.vision_tower.weight, torch.ones(2, 2))
    assert "load hook warning" in caplog.text
    assert "Missing key" not in caplog.text


def test_strict_load_rejects_unexpected_keys(tmp_path):
    model = nn.Linear(2, 2)
    files = write_shards(tmp_path, model.state_dict(), {"extra": torch.ones(1)})
    with pytest.raises(RuntimeError, match="unexpected keys.*extra"):
        PreTrainedPolicy._load_weight_files(model, files, "cpu", True)


def test_checkpoint_reader_inspects_and_loads_all_shards(tmp_path):
    tensors = {"weight": torch.ones(2, 3), "bias": torch.zeros(2)}
    checkpoint = SafetensorsCheckpoint(
        write_shards(tmp_path, {"weight": tensors["weight"]}, {"bias": tensors["bias"]})
    )
    assert checkpoint.keys() == tensors.keys()
    assert checkpoint.shapes() == {k: tuple(v.shape) for k, v in tensors.items()}
    loaded = checkpoint.load_state_dict()
    for key, value in tensors.items():
        torch.testing.assert_close(loaded[key], value)


@pytest.mark.parametrize("weight_map", [{}, {"a": "../model.safetensors"}, {"a": "/model.safetensors"}])
def test_rejects_invalid_shard_index(tmp_path, weight_map):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError):
        safetensors_shard_names(index)


def test_legacy_single_file_hook_requires_migration_for_shards(tmp_path):
    from lerobot.configs import PreTrainedConfig

    calls = []

    class LegacyPolicy(PreTrainedPolicy):
        config_class = PreTrainedConfig
        name = "legacy-test"

        @classmethod
        def _load_as_safetensor(cls, model, model_file, map_location, strict):
            calls.append((model_file, strict))
            return model

    files = write_shards(tmp_path, {"weight": torch.ones(2, 2)}, {"bias": torch.ones(2)})
    model = nn.Linear(2, 2)
    LegacyPolicy._load_weight_files(model, files[:1], "cpu", True)
    assert calls == [(str(files[0]), True)]
    with pytest.raises(NotImplementedError, match="_load_checkpoint"):
        LegacyPolicy._load_weight_files(model, files, "cpu", True)
    assert len(calls) == 1


def test_overriding_both_hooks_loads_shards(tmp_path):
    from lerobot.configs import PreTrainedConfig

    seen = []

    class BothHooksPolicy(PreTrainedPolicy):
        config_class = PreTrainedConfig
        name = "both-hooks"

        @classmethod
        def _load_as_safetensor(cls, model, model_file, map_location, strict):
            seen.append("file")
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        @classmethod
        def _load_checkpoint(cls, model, checkpoint, map_location, strict):
            seen.append(len(checkpoint.files))
            return super()._load_checkpoint(model, checkpoint, map_location, strict)

    model = nn.Linear(2, 2)
    weight, bias = torch.ones(2, 2), torch.ones(2)
    files = write_shards(tmp_path, {"weight": weight}, {"bias": bias})
    loaded = BothHooksPolicy._load_weight_files(model, files, "cpu", True)
    assert seen == [2]
    torch.testing.assert_close(loaded.weight, weight)
    torch.testing.assert_close(loaded.bias, bias)
