#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import inspect
import typing
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
from lerobot.utils.constants import OBS_STATE

ROOT = Path(__file__).resolve().parents[3]


def test_package_init_exports_required_symbols():
    init_source = (ROOT / "src" / "lerobot" / "policies" / "fastwam" / "__init__.py").read_text()

    assert "FastWAMConfig" in init_source
    assert "make_fastwam_pre_post_processors" in init_source


def test_policy_config_is_exported_from_public_policies_package():
    import lerobot.policies as policies

    assert policies.FastWAMConfig is FastWAMConfig
    assert "FastWAMConfig" in policies.__all__


def test_fastwam_policy_docs_are_registered():
    readme_path = ROOT / "src" / "lerobot" / "policies" / "fastwam" / "README.md"
    wan_readme_path = ROOT / "src" / "lerobot" / "policies" / "fastwam" / "wan" / "README.md"
    policy_readme_path = ROOT / "docs" / "source" / "policy_fastwam_README.md"
    guide_path = ROOT / "docs" / "source" / "fastwam.mdx"
    toctree_path = ROOT / "docs" / "source" / "_toctree.yml"

    assert readme_path.is_symlink()
    assert readme_path.resolve() == policy_readme_path.resolve()
    assert wan_readme_path.exists()
    wan_readme = wan_readme_path.read_text()
    assert "Wan-Video/Wan2.2" in wan_readme
    assert "42bf4cfaa384bc21833865abc2f9e6c0e67233dc" in wan_readme
    assert policy_readme_path.exists()
    assert guide_path.exists()
    assert "local: fastwam" in toctree_path.read_text()


def test_wan_backbone_code_is_isolated_from_lerobot_adapter():
    wan_dir = ROOT / "src" / "lerobot" / "policies" / "fastwam" / "wan"

    assert (wan_dir / "modules" / "attention.py").exists()
    assert (wan_dir / "modules" / "model.py").exists()
    assert (wan_dir / "modules" / "t5.py").exists()
    assert (wan_dir / "modules" / "tokenizers.py").exists()
    assert (wan_dir / "modules" / "vae2_1.py").exists()
    assert (wan_dir / "modules" / "vae2_2.py").exists()
    assert (wan_dir / "utils" / "fm_solvers.py").exists()
    assert (wan_dir / "utils" / "fm_solvers_unipc.py").exists()

    assert (wan_dir.parent / "wan_video_dit.py").exists()
    assert (wan_dir.parent / "wan_adapters.py").exists()
    assert (wan_dir.parent / "wan_components.py").exists()
    assert not (wan_dir / "wan_video_dit.py").exists()
    assert not (wan_dir / "wan_adapters.py").exists()
    assert not (wan_dir / "wan_components.py").exists()


def test_fastwam_text_encoder_uses_upstream_wan_modules_directly():
    fastwam_dir = ROOT / "src" / "lerobot" / "policies" / "fastwam"
    modular_source = (fastwam_dir / "modular_fastwam.py").read_text()
    components_source = (fastwam_dir / "wan_components.py").read_text()

    assert not (fastwam_dir / "wan_video_text_encoder.py").exists()
    assert "from .wan.modules.t5 import umt5_xxl" in components_source
    assert "from .wan.modules.tokenizers import HuggingfaceTokenizer" in components_source
    assert "WAN_T5_ENCODER_KWARGS" not in components_source
    assert "wan_video_text_encoder" not in modular_source


def test_fastwam_vae_reuses_upstream_wan_modules():
    fastwam_dir = ROOT / "src" / "lerobot" / "policies" / "fastwam"
    vae_source = (fastwam_dir / "wan_adapters.py").read_text()

    assert not (fastwam_dir / "wan_video_vae.py").exists()
    assert "from .wan.modules.vae2_2 import Wan2_2_VAE" in vae_source
    assert "mean = [" not in vae_source
    assert "std = [" not in vae_source
    assert "class Encoder3d_38" not in vae_source
    assert "class Decoder3d_38" not in vae_source
    assert "class VideoVAE38_" not in vae_source


def test_fastwam_component_loading_uses_fixed_wan_checkpoint_layout():
    modular_source = (ROOT / "src" / "lerobot" / "policies" / "fastwam" / "modular_fastwam.py").read_text()
    modeling_source = (ROOT / "src" / "lerobot" / "policies" / "fastwam" / "modeling_fastwam.py").read_text()
    components_source = (ROOT / "src" / "lerobot" / "policies" / "fastwam" / "wan_components.py").read_text()

    assert "class ModelConfig" not in modular_source
    assert "def load_state_dict" not in modular_source
    assert "WAN22_MODEL_REGISTRY" not in modular_source
    assert "class ModelConfig" not in components_source
    assert "class WanComponentSource" not in components_source
    assert "def load_state_dict" not in components_source
    assert "WAN22_MODEL_REGISTRY" not in components_source
    assert "hash_model_file" not in components_source
    assert "_resolve_component_sources" not in components_source
    assert "origin_file_pattern" not in components_source
    assert "inspect.signature" not in components_source
    assert "class FastWAMWanComponentPaths" not in modeling_source
    assert "def _first_existing" not in modeling_source
    assert "def _missing_wan_component_names" not in modeling_source
    assert "WAN_T5_CHECKPOINT" in components_source
    assert "WAN_VAE_CHECKPOINT" in components_source
    assert "WAN_DIT_PATTERN" in components_source


def test_fastwam_dit_reuses_upstream_wan_primitives():
    dit_source = (ROOT / "src" / "lerobot" / "policies" / "fastwam" / "wan_video_dit.py").read_text()

    assert "from .wan.modules.model import" in dit_source
    assert "WanModel" in dit_source
    for duplicated_symbol in [
        "def flash_attention(",
        "def sinusoidal_embedding_1d(",
        "def rope_apply(",
        "def unpatchify(",
        "def _dense_video_freqs(",
        "class RMSNorm(",
        "class SelfAttention(",
        "class CrossAttention(",
        "class Head(",
    ]:
        assert duplicated_symbol not in dit_source


def test_fastwam_inference_schedule_reuses_upstream_wan_sigmas():
    modular_source = (ROOT / "src" / "lerobot" / "policies" / "fastwam" / "modular_fastwam.py").read_text()

    assert "def _get_wan_sampling_sigmas" in modular_source
    assert "from .wan.utils.fm_solvers import get_sampling_sigmas" in modular_source
    assert "_get_wan_sampling_sigmas(num_inference_steps, shift)" in modular_source


def test_policy_config_rejects_missing_required_image_and_action_features():
    with pytest.raises(ValueError, match="image feature"):
        FastWAMConfig(
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        )

    with pytest.raises(ValueError, match="action"):
        FastWAMConfig(
            output_features={"not_action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


def test_policy_init_calls_validate_features_even_for_prebuilt_configs(monkeypatch):
    cfg = FastWAMConfig(action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)
    calls = []

    def record_validate_features():
        calls.append("called")

    monkeypatch.setattr(cfg, "validate_features", record_validate_features)
    monkeypatch.setattr(
        FastWAMPolicy,
        "_build_core_model",
        lambda self, config: nn.Linear(1, 1),
    )
    FastWAMPolicy(cfg)

    assert calls == ["called"]


def test_required_policy_entrypoints_exist_with_discoverable_names():
    assert FastWAMPolicy.config_class is FastWAMConfig
    assert FastWAMPolicy.name == "fastwam"
    assert callable(FastWAMPolicy.reset)
    assert callable(FastWAMPolicy.get_optim_params)
    assert callable(FastWAMPolicy.predict_action_chunk)
    assert callable(FastWAMPolicy.select_action)
    assert callable(FastWAMPolicy.forward)
    assert callable(make_fastwam_pre_post_processors)
    assert make_fastwam_pre_post_processors.__name__ == "make_fastwam_pre_post_processors"


def test_policy_constructor_and_forward_match_byo_template_contract():
    init_signature = inspect.signature(FastWAMPolicy.__init__)

    assert "dataset_stats" in init_signature.parameters
    assert "core_model" not in init_signature.parameters
    assert typing.get_type_hints(FastWAMPolicy.forward)["return"] == dict[str, torch.Tensor]


def test_saved_config_round_trips_policy_features(tmp_path):
    cfg = FastWAMConfig(action_dim=7, proprio_dim=8, image_size=(224, 448))
    cfg.save_pretrained(tmp_path)

    loaded = FastWAMConfig.from_pretrained(tmp_path)

    assert loaded.type == "fastwam"
    assert loaded.image_features["observation.images.image"].type == FeatureType.VISUAL
    assert loaded.action_feature.shape == (7,)
    assert loaded.robot_state_feature.shape == (8,)


def test_config_from_pretrained_ignores_unknown_fields(tmp_path):
    cfg = FastWAMConfig()
    cfg.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = config_path.read_text()
    payload = payload.replace(
        '"torch_dtype": "bfloat16"',
        '"torch_dtype": "bfloat16",\n  "unknown_fastwam_field": true',
    )
    config_path.write_text(payload)

    loaded = FastWAMConfig.from_pretrained(tmp_path)

    assert loaded.type == "fastwam"
    assert not hasattr(loaded, "unknown_fastwam_field")


def test_config_from_pretrained_does_not_use_non_wan22_tokenizer_repo_id(tmp_path):
    cfg = FastWAMConfig()
    cfg.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = config_path.read_text()
    payload = payload.replace(
        '"tokenizer_model_id": "Wan-AI/Wan2.2-TI2V-5B"',
        '"tokenizer_model_id": "somebody/old-tokenizer"',
    )
    config_path.write_text(payload)

    loaded = FastWAMConfig.from_pretrained(tmp_path)

    assert loaded.tokenizer_model_id == "Wan-AI/Wan2.2-TI2V-5B"
