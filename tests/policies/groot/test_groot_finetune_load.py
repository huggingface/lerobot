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

import logging
import shutil

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import pretrained
from lerobot.policies.groot import modeling_groot
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


def _policy_config(base, device="cpu", model_params_fp32=True):
    return GrootConfig(
        base_model_path=str(base),
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
        device=device,
        model_params_fp32=model_params_fp32,
        chunk_size=4,
        n_action_steps=4,
        # Not the defaults, so a load that drops the model config overrides gives a different model.
        num_inference_timesteps=3,
        tune_top_llm_layers=1,
    )


def _save_finetune(tmp_path, device="cpu", model_params_fp32=True, tune_backbone=True):
    base = _write_tiny_base(tmp_path, tune_backbone)
    policy = GrootPolicy(_policy_config(base, device, model_params_fp32))
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


@pytest.mark.parametrize("model_params_fp32", [True, False])
def test_finetune_matches_regular_load_without_base_weights(tmp_path, monkeypatch, model_params_fp32):
    base, finetune = _save_finetune(tmp_path, model_params_fp32=model_params_fp32)
    expected = _tensors(_load_regular(finetune, monkeypatch))

    (base / "model.safetensors").unlink()
    policy = GrootPolicy.from_pretrained(finetune)
    # The weights must be copies, not views of the checkpoint file.
    _overwrite_weights_in_file(finetune / "model.safetensors")

    _assert_same(expected, _tensors(policy))
    assert not policy.training
    assert policy._groot_model.action_head.num_inference_timesteps == 3
    lm_head, embedding = modeling_groot._TIED_WEIGHT_NAMES
    assert policy._groot_model.get_parameter(lm_head) is policy._groot_model.get_parameter(embedding)


@require_cuda
def test_finetune_matches_regular_load_on_cuda(tmp_path, monkeypatch):
    base, finetune = _save_finetune(tmp_path, device="cuda")
    expected = _tensors(_load_regular(finetune, monkeypatch))

    (base / "model.safetensors").unlink()
    _assert_same(expected, _tensors(GrootPolicy.from_pretrained(finetune)))


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


def test_force_download_downloads_once_and_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)
    forced = []

    def download(repo_id, filename, force_download=False, **kwargs):
        forced.append(force_download)
        return str(finetune / filename)

    monkeypatch.setattr(modeling_groot, "hf_hub_download", download)
    monkeypatch.setattr(pretrained, "hf_hub_download", download)
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.INFO):
        GrootPolicy.from_pretrained(
            "someone/groot-finetune", config=PreTrainedConfig.from_pretrained(finetune), force_download=True
        )
    assert forced == [False, True]
    assert built == [str(base)]
    assert "force_download is set" in caplog.text


def test_mismatched_checkpoint_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)
    from safetensors.torch import load_file, save_file

    weights = load_file(finetune / "model.safetensors")
    weights.pop(next(k for k in weights if "action_head" in k))
    save_file(weights, finetune / "model.safetensors")
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="Missing key"):
        GrootPolicy.from_pretrained(finetune)
    assert built == [None, str(base)]
    assert "keys or shapes differ" in caplog.text


def test_bfloat16_parts_of_the_base_keep_an_fp32_checkpoint_exact(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path, tune_backbone=False)
    expected = _tensors(_load_regular(finetune, monkeypatch))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.INFO):
        policy = GrootPolicy.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert built == [None, str(base)]
    assert "builds some weights in bfloat16" in caplog.text


def test_subclass_loads_the_regular_way(tmp_path, monkeypatch, caplog):
    base, finetune = _save_finetune(tmp_path)

    class TunesLLM(GrootPolicy):
        def __init__(self, config):
            config.tune_llm = True
            super().__init__(config)

    expected = _tensors(_load_regular(finetune, monkeypatch, TunesLLM))
    built = _record_builds(monkeypatch)

    with caplog.at_level(logging.INFO):
        policy = TunesLLM.from_pretrained(finetune)

    _assert_same(expected, _tensors(policy))
    assert built == [str(base)]
    assert "TunesLLM is a subclass" in caplog.text


def test_subclass_weights_are_checked_against_the_checkpoint(tmp_path):
    _, finetune = _save_finetune(tmp_path)

    class WithReference(GrootPolicy):
        def __init__(self, config):
            super().__init__(config)
            self.reference = self._create_groot_model()

    with pytest.raises(RuntimeError, match="Missing key"):
        WithReference.from_pretrained(finetune)
    policy = WithReference.from_pretrained(finetune, strict=False)
    assert policy.reference is not policy._groot_model


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


def test_regular_construction_still_loads_the_base_weights(tmp_path):
    from safetensors.torch import load_file

    base = _write_tiny_base(tmp_path)

    policy = GrootPolicy(_policy_config(base))

    state = policy._groot_model.state_dict()
    base_weights = load_file(base / "model.safetensors")
    assert base_weights
    for name, tensor in base_weights.items():
        assert torch.equal(state[name], tensor.to(state[name].dtype)), name
