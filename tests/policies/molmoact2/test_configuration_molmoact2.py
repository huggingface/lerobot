# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""MolmoAct2 config. It does not need transformers."""

import json
import math

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import MolmoAct2AdamWConfig
from lerobot.policies import get_policy_class, make_policy_config
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERA = "observation.images.top"


def test_config_defaults_follow_the_released_fine_tuning_recipe():
    config = make_policy_config("molmoact2", device="cpu")

    assert isinstance(config, MolmoAct2Config) and get_policy_class("molmoact2") is MolmoAct2Policy
    assert config.type == "molmoact2"
    assert (config.action_mode, config.inference_action_mode) == ("both", "continuous")
    assert (config.chunk_size, config.n_action_steps, config.max_action_dim) == (30, 30, 32)
    assert config.action_delta_indices == list(range(30))
    assert config.observation_delta_indices is None
    # The PI05-style precision and training knobs.
    assert config.dtype == torch.bfloat16
    assert not config.gradient_checkpointing and not config.compile_model
    assert not config.freeze_vision_encoder and not config.train_expert_only
    assert config.freeze_embedding and config.residual_dropout == 0.1
    assert config.normalization_mapping["STATE"] == NormalizationMode.QUANTILES
    assert config.normalization_mapping["ACTION"] == NormalizationMode.QUANTILES
    assert config.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY
    assert config.scheduler_decay_steps is None
    optimizer = config.get_optimizer_preset()
    assert isinstance(optimizer, MolmoAct2AdamWConfig) and optimizer.type == "molmoact2_adamw"
    # The shared trainer's global clipping is off; each component is clipped by optimizer_grad_clip_norm.
    assert (optimizer.grad_clip_norm, optimizer.group_grad_clip_norm) == (
        0.0,
        config.optimizer_grad_clip_norm,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"action_mode": "fast"}, "action_mode must be"),
        ({"inference_action_mode": "both"}, "inference_action_mode must be"),
        ({"action_mode": "discrete"}, "cannot run continuous inference"),
        ({"action_mode": "continuous", "inference_action_mode": "discrete"}, "cannot run discrete inference"),
        ({"train_expert_only": True}, "requires action_mode='continuous'"),
        ({"random_camera_order": "step"}, "random_camera_order must be"),
        ({"n_action_steps": 31}, "cannot exceed chunk_size"),
        ({"joint_offsets": [0.0]}, "requires joint_scales"),
        ({"norm_stats_path": "norm_stats.json"}, "requires norm_tag"),
        ({"dtype": torch.float16}, "dtype must be"),
    ],
)
def test_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MolmoAct2Config(device="cpu", **kwargs)


def test_validate_features_requires_a_camera_and_fills_state_and_action():
    with pytest.raises(ValueError, match="at least one visual input feature"):
        MolmoAct2Config(device="cpu").validate_features()

    config = MolmoAct2Config(
        device="cpu", input_features={CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))}
    )
    config.validate_features()

    assert config.input_features[OBS_STATE].shape == (0,)
    assert config.output_features[ACTION].shape == (32,)


def test_save_drops_the_norm_tag_fields(tmp_path):
    config = MolmoAct2Config(
        device="cpu", norm_tag="libero", norm_stats_path=str(tmp_path / "norm_stats.json")
    )
    config.device = None  # Converted checkpoints leave it unset, for the accelerator available at load time.

    config._save_pretrained(tmp_path)
    reloaded = PreTrainedConfig.from_pretrained(tmp_path)

    # The tag only selects stats when the processors are first built; a saved checkpoint carries its own stats.
    assert isinstance(reloaded, MolmoAct2Config)
    assert reloaded.norm_tag is None and reloaded.norm_stats_path is None
    assert config.norm_tag == "libero"
    assert json.loads((tmp_path / "config.json").read_text())["device"] is None


def test_scheduler_preset_ends_the_cosine_at_the_run_length():
    config = MolmoAct2Config(
        device="cpu", scheduler_warmup_steps=200, optimizer_lr=5e-5, scheduler_alpha_f=0.1
    )
    preset = config.get_scheduler_preset()
    assert preset.num_decay_steps is None

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1.0)
    multiplier = preset.build(optimizer, num_training_steps=600).lr_lambdas[0]

    # LambdaLR index k - 1 is the LR of update k: official MolmoAct2 increments its step before reading the LR.
    assert multiplier(0) == pytest.approx(1 / 200)
    assert multiplier(198) == pytest.approx(0.995)
    # The cosine starts at the end of the warmup, without a jump.
    assert multiplier(199) == pytest.approx(1.0)
    assert multiplier(200) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 400)))
    assert multiplier(399) == pytest.approx(0.55)
    assert multiplier(599) == pytest.approx(0.1)
