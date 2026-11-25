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

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _make_config(image_keys: list[str]) -> VQBeTConfig:
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    for key in image_keys:
        input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96))
    return VQBeTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        crop_shape=(84, 84),
    )


def test_validate_features_accepts_multiple_cameras():
    config = _make_config([f"{OBS_IMAGES}.front", f"{OBS_IMAGES}.wrist"])
    config.validate_features()
    assert len(config.image_features) == 2


def test_validate_features_rejects_zero_cameras():
    config = _make_config([])
    with pytest.raises(ValueError, match="at least one image"):
        config.validate_features()
