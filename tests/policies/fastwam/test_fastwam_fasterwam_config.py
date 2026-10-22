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

"""FastWAM defaults, Faster-WAM option validation and configuration round trips."""

import pytest

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.configs import PreTrainedConfig
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig


def test_default_config_preserves_fastwam_behavior():
    config = FastWAMConfig()
    assert config.action_expert_layers is None
    assert not config.lite_kv_fusion
    assert not config.retracted_1d_rope
    assert not config.world_model_only_conditioning
    assert config.action_dit_config["num_layers"] == 30
    assert (
        "use_text_cross_attention" not in config.action_dit_config
        or config.action_dit_config["use_text_cross_attention"]
    )


def test_fasterwam_config_serialization_round_trip(tmp_path):
    config = FastWAMConfig(
        action_expert_layers=1,
        lite_kv_fusion=True,
        retracted_1d_rope=True,
        world_model_only_conditioning=True,
        action_dit_config={"num_layers": 30},
    )
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path, policy_class="fastwam")
    assert isinstance(loaded, FastWAMConfig)
    assert loaded.action_expert_layers == 1
    assert loaded.lite_kv_fusion
    assert loaded.retracted_1d_rope
    assert loaded.world_model_only_conditioning
    assert loaded.action_dit_config["num_layers"] == 1
    assert loaded.action_dit_config["use_text_cross_attention"] is False


@pytest.mark.parametrize("depth", [2.0, True, "2", 0])
@pytest.mark.parametrize("field", ["action_expert_layers", "action_dit_config"])
def test_action_depth_requires_positive_integer(depth, field):
    kwargs = {field: depth if field == "action_expert_layers" else {"num_layers": depth}}
    with pytest.raises(ValueError, match="positive integer"):
        FastWAMConfig(lite_kv_fusion=True, base_model_id=None, **kwargs)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"action_expert_layers": 1}, "lite_kv_fusion"),
        ({"action_expert_layers": 31, "lite_kv_fusion": True}, "cannot exceed"),
        ({"retracted_1d_rope": True}, "retracted_1d_rope"),
    ],
)
def test_invalid_fasterwam_combinations(options, message):
    with pytest.raises(ValueError, match=message):
        FastWAMConfig(**options)
