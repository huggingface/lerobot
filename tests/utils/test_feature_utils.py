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

from copy import deepcopy

import pytest

from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features


@pytest.mark.parametrize("dtype", ["image", "video"])
@pytest.mark.parametrize(
    "shape,names,expected_shape",
    [
        ((480, 640, 1), None, (1, 480, 640)),
        ((480, 640, 3), None, (3, 480, 640)),
        ((480, 640, 4), None, (4, 480, 640)),
        ((3, 480, 640), None, (3, 480, 640)),
        ((480, 640, 3), ["height", "width", "channel"], (3, 480, 640)),
        ((480, 640, 3), ["height", "width", "channels"], (3, 480, 640)),
        ((3, 480, 640), ["channels", "height", "width"], (3, 480, 640)),
    ],
)
def test_dataset_to_policy_features_visual_layouts(dtype, shape, names, expected_shape):
    features = {
        "observation.images.front": {
            "dtype": dtype,
            "shape": shape,
            "names": names,
        },
    }

    original_features = deepcopy(features)
    policy_features = dataset_to_policy_features(features)

    assert policy_features["observation.images.front"].type is FeatureType.VISUAL
    assert policy_features["observation.images.front"].shape == expected_shape
    assert features == original_features
