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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import datasets

from lerobot.datasets.feature_utils import get_hf_features_from_features


def test_get_hf_features_zero_width_feature_does_not_raise_on_from_dict():
    features = {"empty": {"dtype": "float32", "shape": (0,), "names": ["empty"]}}
    hf_features = get_hf_features_from_features(features)
    datasets.Dataset.from_dict({"empty": [[], []]}, features=hf_features)
