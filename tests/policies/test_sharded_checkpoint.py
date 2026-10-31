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

import importlib
import json
import logging
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

import lerobot.policies.pretrained as pretrained
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.checkpoint import SafetensorsCheckpoint, safetensors_shard_names
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import rename_legacy_vision_model_keys
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from tests.utils import require_cuda


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


def tiny_act_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=4,
        n_action_steps=4,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
    )


class CopiesItsCheckpoint(nn.Linear):
    """What the VLA-JEPA, FineART-VLA and Cosmos3 loaders read of a policy before they copy a checkpoint in."""

    def __init__(self):
        super().__init__(2, 2)
        self.config = SimpleNamespace(reinit_modules=["head."], load_action_modules=False)
        self.model = SimpleNamespace(network=SimpleNamespace(_action_modules=("head",)))

    def _convert_openpi_state_dict(self, state_dict):
        return state_dict


@pytest.mark.parametrize(
    ("module", "name", "extras"),
    [
        ("lerobot.policies.vla_jepa.modeling_vla_jepa", "VLAJEPAPolicy", ("transformers", "diffusers")),
        ("lerobot.policies.fineart_vla.modeling_fineart_vla", "FineARTVLAPolicy", ("transformers",)),
        ("lerobot.policies.cosmos3.modeling_cosmos3", "Cosmos3Policy", ("transformers", "diffusers")),
    ],
)
def test_a_loader_that_copies_its_checkpoint_maps_it_on_the_cpu(tmp_path, monkeypatch, module, name, extras):
    # Parameters built on a GPU, with the checkpoint mapped there too, would hold the weights twice.
    for extra in extras:
        pytest.importorskip(extra)
    policy_class = getattr(importlib.import_module(module), name)
    model = CopiesItsCheckpoint()
    save_file(
        {key: torch.ones_like(value) for key, value in model.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    mapped = []
    read = SafetensorsCheckpoint.load_state_dict

    def load_state_dict(self, device="cpu", **kwargs):
        mapped.append(device)
        return read(self, "cpu", **kwargs)

    monkeypatch.setattr(SafetensorsCheckpoint, "load_state_dict", load_state_dict)
    checkpoint = SafetensorsCheckpoint([tmp_path / "model.safetensors"])
    policy_class._load_checkpoint(model, checkpoint, "meta", True)
    assert mapped == ["cpu"]
    torch.testing.assert_close(model.weight.detach(), torch.ones(2, 2))


def test_a_loader_that_drops_mismatched_weights_leaves_the_policy_where_it_was_built(tmp_path):
    # FastWAM drops the weights whose shape changed with the embodiment; `from_pretrained` moves the policy.
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    model = nn.Linear(2, 2)
    save_file({"weight": torch.ones(3, 2), "bias": torch.ones(2)}, tmp_path / "model.safetensors")
    checkpoint = SafetensorsCheckpoint([tmp_path / "model.safetensors"])
    FastWAMPolicy._load_checkpoint(model, checkpoint, "meta", False)
    assert model.weight.device.type == "cpu"
    torch.testing.assert_close(model.bias.detach(), torch.ones(2))


def test_the_default_loader_maps_every_shard_on_the_cpu(tmp_path, monkeypatch):
    policy = ACTPolicy(tiny_act_config())
    state = policy.state_dict()
    names = sorted(state)
    files = [tmp_path / "model-00001-of-00002.safetensors", tmp_path / "model-00002-of-00002.safetensors"]
    for file, shard in zip(files, (names[::2], names[1::2]), strict=True):
        save_file({name: state[name].clone() for name in shard}, file)
    mapped = []

    def map_shard(file, device):
        mapped.append(device)
        return load_file(file, device="cpu")

    monkeypatch.setattr(pretrained, "load_file", map_shard)
    ACTPolicy._load_checkpoint(policy, SafetensorsCheckpoint(files), "meta", strict=True)
    assert mapped == ["cpu", "cpu"]


def test_a_policy_that_reads_its_checkpoint_itself_gets_its_device(tmp_path, monkeypatch):
    # An override still gets the policy's device (G0.5 maps its file there); the default maps on the CPU.
    config = tiny_act_config()
    ACTPolicy(config).save_pretrained(tmp_path)
    config.device = "meta"
    mapped = {}

    def load_model_as_safetensor(model, model_file, strict, device):
        mapped["file"] = device
        return [], []

    def reads_its_checkpoint(cls, model, checkpoint, map_location, strict):
        mapped["policy"] = map_location
        return PreTrainedPolicy._load_checkpoint.__func__(cls, model, checkpoint, map_location, strict)

    monkeypatch.setattr(pretrained, "load_model_as_safetensor", load_model_as_safetensor)
    monkeypatch.setattr(ACTPolicy, "_load_checkpoint", classmethod(reads_its_checkpoint))
    ACTPolicy.from_pretrained(tmp_path, config=config)
    assert mapped == {"policy": "meta", "file": "cpu"}


class BuiltOnItsDevice(ACTPolicy):
    """Moves itself to its device when built, as pi05 does."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.to(config.device)


@require_cuda
def test_a_policy_built_on_the_gpu_holds_its_weights_once_while_it_loads(tmp_path):
    config = tiny_act_config()
    ACTPolicy(config).save_pretrained(tmp_path)
    saved = load_file(tmp_path / "model.safetensors")
    weights = sum(tensor.numel() * tensor.element_size() for tensor in saved.values())
    config.device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    policy = BuiltOnItsDevice.from_pretrained(tmp_path, config=config)
    # Mapped on the GPU, the whole file would sit there beside the parameters: twice the weights.
    assert torch.cuda.max_memory_allocated() - before < 1.5 * weights
    state = policy.state_dict()
    for key, tensor in saved.items():
        torch.testing.assert_close(state[key].cpu(), tensor)
