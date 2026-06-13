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

from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features


def test_dataset_to_policy_features_handles_visual_names_none_hwc():
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": None,
        },
    }

    policy_features = dataset_to_policy_features(features)

    assert policy_features["observation.images.front"].type is FeatureType.VISUAL
    assert policy_features["observation.images.front"].shape == (3, 480, 640)
