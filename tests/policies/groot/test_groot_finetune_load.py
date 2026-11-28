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

"""A fine-tuned GR00T checkpoint loads without NVIDIA's base weights, with the same result as before."""

import json
import logging
import re
import shutil

import pytest
import torch
from huggingface_hub import constants as hub_constants
from huggingface_hub.utils import filter_repo_objects
from safetensors import safe_open
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import pretrained
from lerobot.policies.groot import groot_n1_7, modeling_groot
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.import_utils import _require_package_cache
from tests.utils import require_cuda


def _write_tiny_base(path, tune_backbone=True):
    from transformers import Qwen3VLConfig

    backbone = path / "backbone"
    Qwen3VLConfig(
        tie_word_embeddings=True,
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 16,
        },
    ).save_pretrained(backbone)
    config = GR00TN17Config(
        model_name=str(backbone),
        select_layer=2,
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=5,
        action_horizon=4,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        vl_self_attention_cfg={"num_layers": 0},
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": False,
        },
        tune_llm=tune_backbone,
        tune_visual=tune_backbone,
        load_bf16=True,
    )
    base = path / "base"
    # Saved in bf16 like NVIDIA's base, so its config asks for bf16 parameters.
    GR00TN17(config, load_backbone_weights=False).to(torch.bfloat16).save_pretrained(base)
    return base


def _policy_config(base, dtype=torch.float32):
    return GrootConfig(
        base_model_path=str(base),
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
        device="cpu",
        dtype=dtype,
        chunk_size=4,
        n_action_steps=4,
        # Not the defaults, so a load that drops the model config overrides gives a different model.
        num_inference_timesteps=3,
        tune_top_llm_layers=1,
    )


def _cap_action_horizon(base, delta_indices):
    action = {"delta_indices": delta_indices}
    processor_kwargs = {"modality_configs": {"new_embodiment": {"action": action}}}
    (base / "processor_config.json").write_text(json.dumps({"processor_kwargs": processor_kwargs}))


def _save_finetune(tmp_path, dtype=torch.float32, tune_backbone=True):
    base = _write_tiny_base(tmp_path, tune_backbone)
    policy = GrootPolicy(_policy_config(base, dtype))
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.is_floating_point():
                noise = torch.randn(parameter.shape, generator=generator).to(parameter.dtype)
                parameter.add_(noise, alpha=1e-3)
    policy.save_pretrained(tmp_path / "finetune")
    return base, tmp_path / "finetune"


def _tensors(policy):
    return {
        name: (tensor.detach().cpu(), tensor.dtype, tensor.device, tensor.requires_grad)
        for name, tensor in [*policy.named_parameters(remove_duplicate=False), *policy.named_buffers()]
    }


def _assert_same(expected, actual):
    assert expected.keys() == actual.keys()
    for name, (tensor, dtype, device, requires_grad) in expected.items():
        other, other_dtype, other_device, other_requires_grad = actual[name]
        assert (dtype, device, requires_grad) == (other_dtype, other_device, other_requires_grad), name
        assert torch.equal(tensor, other), name


def _overwrite_weights_in_file(path):
    with open(path, "r+b") as file:
        header_size = int.from_bytes(file.read(8), "little")
        file.seek(8 + header_size)
        file.write(bytes(path.stat().st_size - 8 - header_size))


def _load_regular(finetune, monkeypatch, policy_cls=GrootPolicy, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(GrootPolicy, "_build_groot_model_from_checkpoint", classmethod(lambda cls, *args: None))
        return policy_cls.from_pretrained(finetune, **kwargs)


def _record_builds(monkeypatch):
    built = []
    real_from_pretrained = GR00TN17.from_pretrained

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        built.append(pretrained_model_name_or_path)
        return real_from_pretrained(pretrained_model_name_or_path, **kwargs)

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(from_pretrained))
    return built


def _name_base(finetune, base_model_path):
    """Make the fine-tune name another base model, without its config."""
    config = json.loads((finetune / "config.json").read_text())
    config.update(base_model_path=base_model_path, base_model_config=None)
    (finetune / "config.json").write_text(json.dumps(config))


def _hub_base(monkeypatch, tmp_path, files, on_download=False):
    """Serves `files` (name to source file) as the Hub repo someone/groot-base, read offline from a cache.
    With on_download, a file enters the cache only when a download asks for it, as on a first run. Returns
    the repo id."""
    commit = "1" * 40
    cache = tmp_path / "hub_base"
    repo = cache / "models--someone--groot-base"

    def download(names):
        snapshot = repo / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(files[name], snapshot / name)
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / "main").write_text(commit)
        return snapshot

    def snapshot_download(repo_id, allow_patterns=None, **kwargs):
        return str(download(filter_repo_objects(files, allow_patterns=allow_patterns)))

    cache.mkdir()
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    if on_download:
        monkeypatch.setattr(groot_n1_7, "snapshot_download", snapshot_download)
        monkeypatch.setattr(modeling_groot, "snapshot_download", snapshot_download)
    else:
        download(files)
    return "someone/groot-base"


def test_finetune_matches_regular_load_without_base_weights(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path, dtype=torch.bfloat16)
    regular = _load_regular(finetune, monkeypatch)
    assert regular._groot_model.action_head.num_inference_timesteps == 3
    expected = _tensors(regular)

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune)
    # The weights must be copies, not views of the checkpoint file.
    _overwrite_weights_in_file(finetune / "model.safetensors")

    _assert_same(expected, _tensors(policy))
    assert not policy.training
    assert policy._groot_model.action_head.num_inference_timesteps == 3
    lm_head, embedding = modeling_groot._TIED_WEIGHT_NAMES
    assert policy._groot_model.get_parameter(lm_head) is policy._groot_model.get_parameter(embedding)
    # Both loads share the code that applies tune_top_llm_layers, so check its effect directly.
    first, top = policy._groot_model.backbone.language_model.layers
    assert all(parameter.requires_grad for parameter in top.parameters())
    assert not any(parameter.requires_grad for parameter in first.parameters())


@require_cuda
def test_finetune_loads_onto_the_configured_device(tmp_path):
    base, finetune = _save_finetune(tmp_path)
    config = PreTrainedConfig.from_pretrained(finetune)
    config.device = "cuda"

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune, config=config)

    assert {tensor.device.type for tensor in [*policy.parameters(), *policy.buffers()]} == {"cuda"}


def test_loading_leaves_the_global_random_state_as_it_was(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    state = torch.get_rng_state()

    # The regular load draws random numbers to initialize the weights that the checkpoint then replaces.
    _load_regular(finetune, monkeypatch)

    assert torch.equal(torch.get_rng_state(), state)


def test_finetune_loads_by_hub_repo_id(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    # A Hugging Face cache that holds the fine-tune, so it loads by repo id without a download.
    commit = "0" * 40
    repo = tmp_path / "hub" / "models--someone--groot-finetune"
    shutil.copytree(finetune, repo / "snapshots" / commit)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)

    shutil.rmtree(base)
    policy = GrootPolicy.from_pretrained(
        "someone/groot-finetune", cache_dir=tmp_path / "hub", local_files_only=True
    )

    _assert_same(expected, _tensors(policy))


def test_regular_load_builds_from_the_saved_base_config(tmp_path):
    base, finetune = _save_finetune(tmp_path)
    # The base changes after the save, so a load that reads it instead builds another model.
    model_config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    config = GrootConfig.from_pretrained(finetune)
    # As the train config of a run from the base model holds it.
    config.pretrained_path = None

    class Subclass(GrootPolicy):
        pass

    policy = Subclass.from_pretrained(finetune, config=config)
    policy.save_pretrained(tmp_path / "resaved")

    assert policy._groot_model.config.action_horizon == 4
    assert config.pretrained_path is None
    # A save after the load keeps the saved config, not the changed base.
    assert GrootPolicy.from_pretrained(tmp_path / "resaved")._groot_model.config.action_horizon == 4


def test_finetune_loads_without_the_base_model(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    # Caps the action horizon at 2 steps, so a load that loses it gives 4.
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    finetune = tmp_path / "finetune"
    regular = _load_regular(finetune, monkeypatch)
    assert regular._action_queue_steps == 2
    expected = _tensors(regular)

    shutil.rmtree(base)
    policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert policy._action_queue_steps == 2
    assert policy.chunk_inference_spec().prediction_steps == 2
    assert policy.config.action_delta_indices == [0, 1]
    assert policy.config.drop_n_last_frames == 1
    # Saving again keeps the fine-tune self-contained, also over the folder it was loaded from or after
    # that folder is gone.
    policy.save_pretrained(finetune)
    assert GrootPolicy.from_pretrained(finetune)._action_queue_steps == 2
    shutil.rmtree(finetune)
    policy.save_pretrained(tmp_path / "resaved")
    assert GrootPolicy.from_pretrained(tmp_path / "resaved")._action_queue_steps == 2


def test_checkpoint_saved_before_the_base_config_loads_from_the_base_model(tmp_path):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    finetune = tmp_path / "finetune"
    expected = _tensors(GrootPolicy.from_pretrained(finetune))
    # As older versions saved it, without either key.
    config = json.loads((finetune / "config.json").read_text())
    del config["embodiment_action_horizons"], config["base_model_config"]
    (finetune / "config.json").write_text(json.dumps(config))

    # It reads only the base config.json, not the base weights.
    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert policy._action_queue_steps == 2
    assert policy.config.action_delta_indices == [0, 1]


def test_unreadable_base_config_of_an_older_checkpoint_falls_back_to_the_regular_load(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    expected = _tensors(GrootPolicy.from_pretrained(finetune))
    files = {name: base / name for name in ("config.json", "model.safetensors")}
    _name_base(finetune, _hub_base(monkeypatch, tmp_path, files))

    # Offline, the base model is only in the default cache, from which the regular load builds it.
    policy = GrootPolicy.from_pretrained(finetune, cache_dir=tmp_path / "empty_cache")

    _assert_same(expected, _tensors(policy))


def test_saved_horizons_follow_the_embodiment_tag(tmp_path):
    base = _write_tiny_base(tmp_path)
    steps = {"new_embodiment": 2, "other": 1}
    modality_configs = {tag: {"action": {"delta_indices": list(range(n))}} for tag, n in steps.items()}
    processor_config = {"processor_kwargs": {"modality_configs": modality_configs}}
    (base / "processor_config.json").write_text(json.dumps(processor_config))
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "finetune")
    # The base model now gives a horizon for new_embodiment only.
    _cap_action_horizon(base, [0, 1, 2])
    # As lerobot-train reads --policy.path=<checkpoint> --policy.embodiment_tag=other.
    config = PreTrainedConfig.from_pretrained(tmp_path / "finetune", cli_overrides=["--embodiment_tag=other"])

    assert config.action_delta_indices == [0]
    policy = GrootPolicy.from_pretrained(tmp_path / "finetune", config=config)
    assert policy._action_queue_steps == 1
    policy.save_pretrained(tmp_path / "resaved")
    assert GrootConfig.from_pretrained(tmp_path / "resaved").action_delta_indices == [0]


def test_loading_a_base_model_builds_from_its_config(tmp_path):
    base, finetune = _save_finetune(tmp_path)
    # Holds the base config saved with the checkpoint, and names it, as lerobot-train's --policy.path does.
    config = GrootConfig.from_pretrained(finetune)
    config.pretrained_path = finetune
    # Another base model, in shards like NVIDIA's, so loading it by path takes the base model branch.
    other_base = shutil.copytree(base, tmp_path / "other_base")
    model_config = json.loads((other_base / "config.json").read_text())
    (other_base / "config.json").write_text(json.dumps({**model_config, "action_horizon": 2}))
    shard = (other_base / "model.safetensors").rename(other_base / "model-00001-of-00001.safetensors")

    with safe_open(shard, framework="pt") as weights:
        index = {"metadata": {}, "weight_map": dict.fromkeys(weights.keys(), shard.name)}
    (other_base / "model.safetensors.index.json").write_text(json.dumps(index))

    policy = GrootPolicy.from_pretrained(other_base, config=config)

    assert policy._groot_model.config.action_horizon == 2


@pytest.mark.parametrize("base_horizons", [[0], None], ids=["changed", "gone"])
def test_resume_gives_the_dataset_and_the_policy_the_same_horizons(tmp_path, base_horizons):
    base = _write_tiny_base(tmp_path)
    _cap_action_horizon(base, [0, 1])
    GrootPolicy(_policy_config(base)).save_pretrained(tmp_path / "checkpoint")
    if base_horizons is None:
        (base / "processor_config.json").unlink()
    else:
        _cap_action_horizon(base, base_horizons)
    steps = len(base_horizons or [0, 1])

    # As lerobot-train resumes a run from the base model: it builds the dataset with pretrained_path unset,
    # then make_policy points it at the checkpoint.
    config = GrootConfig.from_pretrained(tmp_path / "checkpoint")
    config.pretrained_path = None
    assert len(config.action_delta_indices) == steps
    config.pretrained_path = tmp_path / "checkpoint"
    assert GrootPolicy(config)._action_queue_steps == steps


def test_hub_base_model_saves_its_config(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    files = {name: base / name for name in ("config.json", "model.safetensors")}
    policy = GrootPolicy(_policy_config(_hub_base(monkeypatch, tmp_path, files, on_download=True)))
    policy.save_pretrained(tmp_path / "finetune")
    shutil.rmtree(tmp_path / "hub_base")
    shutil.rmtree(base)

    _assert_same(_tensors(policy), _tensors(GrootPolicy.from_pretrained(tmp_path / "finetune")))


@pytest.mark.parametrize("subclass", [False, True])
def test_force_download_downloads_each_file_once(tmp_path, monkeypatch, subclass):
    base, finetune = _save_finetune(tmp_path)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    forced = []

    def download(repo_id, filename, force_download=False, **kwargs):
        forced.append(force_download)
        return str(finetune / filename)

    class Subclass(GrootPolicy):
        pass

    monkeypatch.setattr(modeling_groot, "hf_hub_download", download)
    monkeypatch.setattr(pretrained, "hf_hub_download", download)
    built = _record_builds(monkeypatch)
    if not subclass:
        (base / "model.safetensors").unlink()

    policy = (Subclass if subclass else GrootPolicy).from_pretrained(
        "someone/groot-finetune", config=PreTrainedConfig.from_pretrained(finetune), force_download=True
    )

    # The check for the weights file, the weights file, and for the regular load its second look for it.
    assert forced == ([False, True, False] if subclass else [False, True])
    assert built == [str(base) if subclass else None]
    _assert_same(expected, _tensors(policy))


def test_failed_forced_download_reaches_the_caller(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    def download(repo_id, filename, force_download=False, **kwargs):
        if force_download:
            raise OSError("No space left on device")
        return str(finetune / filename)

    monkeypatch.setattr(modeling_groot, "hf_hub_download", download)

    # Raised, rather than taking the fine-tune for a base model.
    with pytest.raises(OSError, match="No space left on device"):
        GrootPolicy.from_pretrained("someone/groot-finetune", force_download=True)


def test_checkpoint_with_only_the_embedding_name_needs_no_base_weights(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head, embedding = (f"_groot_model.{name}" for name in modeling_groot._TIED_WEIGHT_NAMES)
    weights[embedding] = weights.pop(lm_head)
    save_file(weights, finetune / "model.safetensors")
    expected = _tensors(_load_regular(finetune, monkeypatch))

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))


def test_invalid_features_are_reported_before_building(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    config = GrootConfig.from_pretrained(finetune)
    config.input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(config.max_state_dim + 1,)
    )
    built = _record_builds(monkeypatch)

    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        GrootPolicy.from_pretrained(finetune, config=config)
    assert built == []


def test_wrong_sized_checkpoint_is_rejected_without_the_base_weights(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    key = next(k for k in weights if "action_head" in k and weights[k].dim() == 2)
    weights[key] = torch.cat([weights[key], weights[key][:1]])
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(RuntimeError, match="size mismatch") as error:
        GrootPolicy.from_pretrained(finetune, strict=False)
    assert built == [None]
    # The regular load raises the same error, after it reads the base weights.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


def test_checkpoint_missing_a_weight_loads_the_regular_way_when_not_strict(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    weights.pop(next(k for k in weights if "action_head" in k))
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    # The base weights fill the missing one.
    with caplog.at_level(logging.WARNING):
        GrootPolicy.from_pretrained(finetune, strict=False)
    assert built == [None, str(base)]
    assert "keys differ" in caplog.text


def test_checkpoint_without_the_tied_weight_is_rejected_in_the_same_words(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    # A save keeps one name of the tied LM head and embedding, so this drops both.
    weights.pop(f"_groot_model.{modeling_groot._TIED_WEIGHT_NAMES[0]}")
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
    assert built == [None]
    # The regular load raises the same error, which names the tied weight once.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


@pytest.mark.parametrize(
    "faults",
    [
        "extra_and_missing",
        "two_wrong_sizes",
        "wrong_sized_lm_head",
        "both_tied_names_wrong_sized_embedding",
    ],
)
def test_faults_are_reported_in_the_order_and_names_of_the_regular_load(tmp_path, monkeypatch, faults):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head, embedding = (f"_groot_model.{name}" for name in modeling_groot._TIED_WEIGHT_NAMES)
    head = next(k for k in weights if "action_head" in k and weights[k].dim() == 2)
    backbone = next(k for k in weights if "backbone" in k and weights[k].dim() == 2 and k != lm_head)

    def grow(key, source=None):
        tensor = weights[source or key]
        weights[key] = torch.cat([tensor, tensor[:1]])

    if faults == "extra_and_missing":
        weights["extra.weight"] = torch.zeros(3)
        weights["_groot_model.extra.weight"] = torch.zeros(3)
        weights.pop(head)
    elif faults == "two_wrong_sizes":
        grow(head)
        grow(backbone)
    elif faults == "wrong_sized_lm_head":
        grow(lm_head)
    else:
        grow(embedding, source=lm_head)
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
    assert built == [None]
    # The regular load raises the same error: torch reports weights of another shape first, in the model's
    # order and under the names the checkpoint holds them, and safetensors lists missing and extra weights
    # together.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


def test_checkpoint_with_both_tied_names_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head = weights["_groot_model.backbone.model.lm_head.weight"].clone()
    weights["_groot_model.backbone.model.model.language_model.embed_tokens.weight"] = lm_head + 1
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune, strict=False)
        assert torch.equal(policy.get_parameter("_groot_model.backbone.model.lm_head.weight"), lm_head)
    assert built == [str(base)]
    assert "weights that GR00T lacks" in caplog.text


def test_bfloat16_parts_of_the_base_keep_an_fp32_checkpoint_exact(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path, tune_backbone=False)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert built == [None, str(base)]
    assert "builds some weights in bfloat16" in caplog.text


def test_base_config_without_dtype_loads_bfloat16_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)
    config = GrootConfig.from_pretrained(finetune)
    config.base_model_config["dtype"] = None
    config.dtype = torch.bfloat16
    expected = _tensors(_load_regular(finetune, monkeypatch, config=config))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = GrootPolicy.from_pretrained(finetune, config=config)

    _assert_same(expected, _tensors(policy))
    assert built == [str(base)]
    assert "does not name a dtype" in caplog.text


def test_subclass_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)

    class TunesLLM(GrootPolicy):
        def __init__(self, config):
            config.tune_llm = True
            super().__init__(config)

    expected = _tensors(_load_regular(finetune, monkeypatch, TunesLLM))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.WARNING):
        policy = TunesLLM.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert built == [str(base)]
    assert "TunesLLM is a subclass" in caplog.text


def test_missing_groot_extra_gives_the_install_hint(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    # Without transformers, the package check fails and the config class has no loader.
    monkeypatch.setitem(_require_package_cache, "transformers", False)
    monkeypatch.setattr(modeling_groot, "GR00TN17Config", type("GR00TN17Config", (), {}))

    with pytest.raises(ImportError, match=r"lerobot\[groot\]"):
        GrootPolicy.from_pretrained(finetune)


def test_unset_device_is_reported_before_building(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    config = GrootConfig.from_pretrained(finetune)
    config.device = None
    built = _record_builds(monkeypatch)

    with pytest.raises(ValueError, match="device is unset"):
        GrootPolicy.from_pretrained(finetune, config=config)
    assert built == []


def test_hub_options_reach_the_base_config(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    files = {name: base / name for name in ("config.json", "model.safetensors")}
    _name_base(finetune, _hub_base(monkeypatch, tmp_path, files))
    # Without strict, a missing weight makes the load fall back to the regular one, which builds the model
    # from the base weights.
    weights = load_file(finetune / "model.safetensors")
    weights.pop(next(key for key in weights if "action_head" in key))
    save_file(weights, finetune / "model.safetensors")
    download = modeling_groot.snapshot_download
    options = []

    def recording_download(repo_id, **kwargs):
        options.append(
            (repo_id, kwargs.get("cache_dir"), kwargs.get("local_files_only"), kwargs.get("token"))
        )
        return download(repo_id, **kwargs)

    monkeypatch.setattr(modeling_groot, "snapshot_download", recording_download)
    GrootPolicy.from_pretrained(
        finetune, cache_dir=tmp_path / "hub_base", local_files_only=True, token="hf_test", strict=False
    )

    # So the config.json of a base model given by Hub id is read once, with the caller's cache, cache only and
    # token options, also when the load falls back to the regular one.
    assert options == [("someone/groot-base", tmp_path / "hub_base", True, "hf_test")]


def test_loader_options_reach_the_model(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    config = _policy_config(base)
    # None of these are the defaults of GR00TN17.from_pretrained, and only tune_visual is GrootConfig's.
    config.tune_llm = True
    config.tune_visual = False
    config.tune_projector = False
    config.tune_diffusion_model = False
    config.tune_vlln = False
    config.rtc_ramp_rate = 2.5
    built = _record_builds(monkeypatch)

    policy = GrootPolicy.from_pretrained(finetune, config=config)

    assert built == [None]
    model = policy._groot_model
    assert all(parameter.requires_grad for parameter in model.backbone.language_model.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.visual.parameters())
    for module in (model.action_head.state_encoder, model.action_head.model, model.action_head.vlln):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert model.config.rtc_ramp_rate == 2.5


def test_a_local_config_that_is_not_groot_is_refused_before_building(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    # Missing keys take the N1.7 defaults, so any JSON object builds a GR00T. This one builds the tiny one,
    # as credentials would next to a real fine-tune, but names no GR00T model.
    model_config = json.loads((base / "config.json").read_text())
    del model_config["model_type"]
    secret = tmp_path / "secret" / "config.json"
    secret.parent.mkdir()
    secret_config = {**model_config, "action_horizon": 2, "auths": {"example": {"auth": "SECRET"}}}
    secret.write_text(json.dumps(secret_config))
    _name_base(finetune, str(secret.parent))
    # With base weights, so a load that read that config.json to build a model could finish.
    shutil.copyfile(base / "model.safetensors", secret.parent / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(ValueError, match="is not a GR00T N1.7 model"):
        GrootPolicy.from_pretrained(finetune)
    assert built == []


def test_hub_base_config_that_is_not_groot_is_refused_before_its_weights_download(tmp_path, monkeypatch):
    base = _write_tiny_base(tmp_path)
    model_config = json.loads((base / "config.json").read_text())
    del model_config["model_type"]
    served = tmp_path / "served.json"
    served.write_text(json.dumps(model_config))
    files = {"config.json": served, "model.safetensors": base / "model.safetensors"}
    built = _record_builds(monkeypatch)

    # A first run from it: nothing is cached yet.
    with pytest.raises(ValueError, match="is not a GR00T N1.7 model"):
        GrootPolicy(_policy_config(_hub_base(monkeypatch, tmp_path, files, on_download=True)))
    assert built == []
    assert not list((tmp_path / "hub_base").rglob("model.safetensors"))


def test_hub_base_config_that_cannot_be_downloaded_is_left_to_the_build(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "empty_cache"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    built = _record_builds(monkeypatch)

    config = _policy_config("someone/groot-base")
    # As a config saved with a checkpoint of another base model holds it.
    config.base_model_config = {"model_type": "Gr00tN1d7"}

    with caplog.at_level(logging.WARNING), pytest.raises(OSError):
        GrootPolicy(config)
    assert built == ["someone/groot-base"]
    assert config.base_model_config is None
    # After a passing network error the build can succeed, and its checkpoints then need the base model.
    assert "need the base model to load" in caplog.text


def test_finetune_without_a_base_config_or_its_base_model_fails_the_regular_way(tmp_path, caplog):
    _, finetune = _save_finetune(tmp_path)
    # A base model on another machine, which is not a repo id either.
    _name_base(finetune, str(tmp_path / "elsewhere" / "base"))

    with caplog.at_level(logging.WARNING), pytest.raises(OSError):
        GrootPolicy.from_pretrained(finetune)
    assert "no readable config.json" in caplog.text


@pytest.mark.parametrize(
    ("consent", "config_consent"), [(False, None), (True, None), (False, True), (True, False)]
)
def test_saved_base_config_runs_backbone_custom_code_only_with_consent(tmp_path, consent, config_consent):
    base, finetune = _save_finetune(tmp_path)
    # A backbone folder whose config asks transformers to run a module that leaves a mark when imported. Its
    # name is unique, as transformers imports a module of the same folder name only once per process.
    backbone = shutil.copytree(base.parent / "backbone", tmp_path / f"backbone_{tmp_path.name}")
    mark = tmp_path / "custom_code_ran"
    (backbone / "custom_config.py").write_text(
        f"from transformers import Qwen3VLConfig\nopen({str(mark)!r}, 'w').close()\n"
        "class CustomConfig(Qwen3VLConfig):\n    pass\n"
    )
    backbone_config = json.loads((backbone / "config.json").read_text())
    backbone_config["auto_map"] = {"AutoConfig": "custom_config.CustomConfig"}
    (backbone / "config.json").write_text(json.dumps(backbone_config))
    finetune_config = json.loads((finetune / "config.json").read_text())
    finetune_config["base_model_config"]["model_name"] = str(backbone)
    (finetune / "config.json").write_text(json.dumps(finetune_config))

    # The caller's consent wins over a given config's.
    config = None
    if config_consent is not None:
        config = PreTrainedConfig.from_pretrained(finetune, trust_remote_code=config_consent)
    GrootPolicy.from_pretrained(finetune, config=config, trust_remote_code=consent)
    # The caller's config is left as it was.
    assert config is None or config.trust_remote_code is config_consent

    assert mark.exists() is consent
