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
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import pretrained
from lerobot.policies.groot import modeling_groot
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.import_utils import _require_package_cache


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


def _load_regular(finetune, monkeypatch, policy_cls=GrootPolicy):
    with monkeypatch.context() as patch:
        patch.setattr(GrootPolicy, "_build_groot_model_from_checkpoint", classmethod(lambda cls, *args: None))
        return policy_cls.from_pretrained(finetune)


def _record_builds(monkeypatch):
    built = []
    real_from_pretrained = GR00TN17.from_pretrained

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        built.append(pretrained_model_name_or_path)
        return real_from_pretrained(pretrained_model_name_or_path, **kwargs)

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(from_pretrained))
    return built


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

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(
        "someone/groot-finetune", cache_dir=tmp_path / "hub", local_files_only=True
    )

    _assert_same(expected, _tensors(policy))


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

    # As on main, rather than taking the fine-tune for a base model.
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


def test_checkpoint_with_weights_outside_groot_is_rejected_without_the_base_weights(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    lm_head, embedding = modeling_groot._TIED_WEIGHT_NAMES
    # Outside GR00T, under the name that a GR00T weight has without its prefix.
    weights[lm_head] = weights[f"_groot_model.{lm_head}"] + 1
    # And both names of the tied pair, so the error lists two weights.
    weights[f"_groot_model.{embedding}"] = weights[f"_groot_model.{lm_head}"].clone()
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
    assert built == [None]
    # The regular load raises the same error, after it reads the base weights.
    with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
        _load_regular(finetune, monkeypatch)


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


@pytest.mark.parametrize("strict", [True, False])
def test_checkpoint_missing_a_weight_loads_the_regular_way_only_when_not_strict(
    tmp_path, monkeypatch, caplog, strict
):
    base, finetune = _save_finetune(tmp_path)

    weights = load_file(finetune / "model.safetensors")
    weights.pop(next(k for k in weights if "action_head" in k))
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    if strict:
        with pytest.raises(RuntimeError) as error:
            GrootPolicy.from_pretrained(finetune)
        assert built == [None]
        # The regular load raises the same error, after it reads the base weights.
        with pytest.raises(RuntimeError, match=f"^{re.escape(str(error.value))}$"):
            _load_regular(finetune, monkeypatch)
    else:
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

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
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
        weights.pop(head)
    elif faults == "two_wrong_sizes":
        grow(head)
        grow(backbone)
    elif faults == "wrong_sized_lm_head":
        grow(lm_head)
    else:
        grow(embedding, source=lm_head)
    save_file(weights, finetune / "model.safetensors")

    with pytest.raises(RuntimeError) as error:
        GrootPolicy.from_pretrained(finetune)
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
    config_file = base / "config.json"
    config_file.write_text(json.dumps({**json.loads(config_file.read_text()), "dtype": None}))
    config = GrootConfig.from_pretrained(finetune)
    config.dtype = torch.bfloat16
    with monkeypatch.context() as patch:
        patch.setattr(GrootPolicy, "_build_groot_model_from_checkpoint", classmethod(lambda cls, *args: None))
        expected = _tensors(GrootPolicy.from_pretrained(finetune, config=config))
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


def test_unset_device_is_reported(tmp_path):
    _, finetune = _save_finetune(tmp_path)
    config = GrootConfig.from_pretrained(finetune)
    config.device = None

    with pytest.raises(ValueError, match="device is unset"):
        GrootPolicy.from_pretrained(finetune, config=config)


def test_hub_options_reach_the_base_config(tmp_path, monkeypatch):
    _, finetune = _save_finetune(tmp_path)
    read = GR00TN17Config.from_pretrained
    options = []

    def from_pretrained(cls, path, **kwargs):
        options.append((kwargs.get("cache_dir"), kwargs.get("local_files_only"), kwargs.get("token")))
        return read(path, **kwargs)

    monkeypatch.setattr(GR00TN17Config, "from_pretrained", classmethod(from_pretrained))
    GrootPolicy.from_pretrained(finetune, cache_dir=tmp_path / "hub", local_files_only=True, token="hf_test")

    # So a base model given by Hub id is read with the caller's cache, cache only or token options.
    assert options == [(tmp_path / "hub", True, "hf_test")]


def test_loader_options_reach_the_model(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path)
    config = _policy_config(base)
    # None of these are the defaults, nor the defaults of GR00TN17.from_pretrained.
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
