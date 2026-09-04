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

"""Guards against silent recipe contamination when creating or loading configs.

The port's historical defaults (MSE ``fm`` loss, frozen ViT, bidirectional
prefix) silently diverged from the official RoboTwin SFT recipe on every
config created from scratch. These tests pin the corrected defaults and the
official-recipe preset helper.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_defaults_match_official_recipe():
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

    config = LingbotVLAV2Config()
    assert config.loss_type == "L1_fm"
    assert config.freeze_vision_encoder is False
    assert config.vlm_causal is True


def test_as_official_recipe_overrides_legacy_checkpoint_values():
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

    legacy = LingbotVLAV2Config(loss_type="fm", freeze_vision_encoder=True, vlm_causal=False)
    recipe = legacy.as_official_recipe()
    assert recipe.loss_type == "L1_fm"
    assert recipe.freeze_vision_encoder is False
    assert recipe.vlm_causal is True
    assert recipe.optimizer_lr == 1e-4
    assert recipe.scheduler_decay_lr == 5e-5
    assert recipe.scheduler_warmup_steps == 0
    # The source object is not mutated.
    assert legacy.loss_type == "fm"


def test_converter_overrides_match_official_recipe():
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config
    from lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint import (
        UPSTREAM_CONFIG_OVERRIDES,
    )

    recipe_keys = {"loss_type", "freeze_vision_encoder", "vlm_causal"}
    assert recipe_keys <= set(UPSTREAM_CONFIG_OVERRIDES)
    config = LingbotVLAV2Config()
    for key in recipe_keys:
        assert UPSTREAM_CONFIG_OVERRIDES[key] == getattr(config, key), (
            f"converter override for {key} drifted from the official-recipe default"
        )


def test_saved_checkpoint_embeds_recipe_not_local_paths(tmp_path):
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

    config = LingbotVLAV2Config(
        robot_config_path="/tmp/some_machine/robot.yaml",
        norm_stats_path="/tmp/some_machine/stats.json",
        robot_config={"joints": "arm.position: 6", "norm_stats": "/tmp/some_machine/stats.json"},
        norm_stats={"norm_stats": {"action.arm.position": {"mean": [0.0], "std": [1.0]}}},
    )
    config.save_pretrained(tmp_path)
    reloaded = LingbotVLAV2Config.from_pretrained(tmp_path)

    assert reloaded.robot_config_path is None
    assert reloaded.norm_stats_path is None
    assert "norm_stats" not in reloaded.robot_config
    # Embedded contents survive; the recipe values are not disturbed by saving.
    assert reloaded.loss_type == "L1_fm"
    assert reloaded.norm_stats is not None
