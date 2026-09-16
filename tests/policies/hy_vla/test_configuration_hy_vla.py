#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import HyVLAConfig
from lerobot.policies.factory import get_policy_class, make_policy_config


def test_hy_vla_factory_registration():
    config = make_policy_config("hy_vla", device="cpu")
    assert isinstance(config, HyVLAConfig)
    pytest.importorskip("transformers", reason="Loading HyVLAPolicy requires the `hy_vla` extra")
    assert get_policy_class("hy_vla").__name__ == "HyVLAPolicy"


def test_umi_contract():
    config = HyVLAConfig(device="cpu")
    assert config.chunk_size == 50
    assert config.physical_action_horizon == 50
    assert config.num_steps == 10
    assert config.max_state_dim == config.max_action_dim == 32


def test_robotwin_rel_abs_contract():
    config = HyVLAConfig(
        device="cpu",
        chunk_size=40,
        n_action_steps=40,
        action_representation="relative_absolute",
        action_decode_mode="blend",
        embodiment="robotwin_dual_arm",
        native_quaternion_order="wxyz",
        use_video_encoder=True,
        img_history_size=6,
        img_history_interval=5,
        execution_horizon=7,
    )
    assert config.physical_action_horizon == 20
    assert config.execution_horizon == 7
    assert config.action_delta_indices == list(range(20))


def test_unnamed_mobile_action_is_rejected():
    with pytest.raises(ValueError, match="umi_dual_arm action"):
        HyVLAConfig(
            device="cpu",
            output_features={"action": PolicyFeature(FeatureType.ACTION, (12,))},
        )
