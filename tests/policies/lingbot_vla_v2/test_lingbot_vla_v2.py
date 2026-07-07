# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Tests for the LingBot-VLA 2.0 (``lingbot_vla_v2``) policy.

The config-level tests here are CUDA-agnostic and run in the base CI. Weight-loading
and forward/inference tests (added with the modeling milestone) are guarded by
``importorskip`` and ``@require_cuda``.
"""

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.utils.constants import ACTION, OBS_STATE


def test_config_creation():
    """The factory returns a registered ``LingbotVLAV2Config`` for ``lingbot_vla_v2``."""
    config = make_policy_config("lingbot_vla_v2")
    assert type(config).__name__ == "LingbotVLAV2Config"
    assert config.type == "lingbot_vla_v2"


def test_config_defaults_match_v2_canonical():
    """Defaults track the upstream LingBot-VLA 2.0 canonical setup."""
    config = make_policy_config("lingbot_vla_v2")
    assert config.max_state_dim == 55
    assert config.max_action_dim == 55
    assert config.chunk_size == 50
    assert config.vlm_family == "qwen3_vl"
    assert config.tokenizer_path == "Qwen/Qwen3-VL-4B-Instruct"
    # MoE + native-resolution image tokens are the defining v2 additions.
    assert config.return_image_grid_thw is True
    assert config.action_num_attention_heads == 32
    assert config.action_num_key_value_heads == 8


def test_kwargs_override():
    config = make_policy_config(
        "lingbot_vla_v2", use_moe=True, max_state_dim=75, max_action_dim=75, num_steps=4
    )
    assert config.use_moe is True
    assert config.max_state_dim == 75
    assert config.num_steps == 4


def test_validate_features_adds_state_and_action():
    """``validate_features`` injects canonical state/action features when absent."""
    config = make_policy_config("lingbot_vla_v2")
    config.input_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    config.output_features = {}
    config.validate_features()
    assert OBS_STATE in config.input_features
    assert config.input_features[OBS_STATE].shape == (config.max_state_dim,)
    assert ACTION in config.output_features
    assert config.output_features[ACTION].shape == (config.max_action_dim,)


def test_optimizer_and_scheduler_presets():
    config = make_policy_config("lingbot_vla_v2")
    optimizer = config.get_optimizer_preset()
    scheduler = config.get_scheduler_preset()
    assert optimizer.lr == config.optimizer_lr
    assert scheduler.num_warmup_steps == config.scheduler_warmup_steps
