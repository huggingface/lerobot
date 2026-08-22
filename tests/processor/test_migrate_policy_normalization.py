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

"""
Tests for the policy normalization migration script.
"""

from lerobot.processor.migrate_policy_normalization import coerce_list_fields_to_tuples


def test_coerce_list_fields_to_tuples_converts_plain_lists():
    cleaned_config = {"crop_shape": [84, 84], "down_dims": [512, 1024, 2048]}

    result = coerce_list_fields_to_tuples(cleaned_config)

    assert result["crop_shape"] == (84, 84)
    assert result["down_dims"] == (512, 1024, 2048)


def test_coerce_list_fields_to_tuples_skips_feature_dicts():
    features = {"observation.image": object()}
    cleaned_config = {"input_features": features, "output_features": features}

    result = coerce_list_fields_to_tuples(cleaned_config)

    assert result["input_features"] is features
    assert result["output_features"] is features


def test_coerce_list_fields_to_tuples_leaves_non_list_values_untouched():
    cleaned_config = {"horizon": 16, "vision_backbone": "resnet18", "crop_shape": None}

    result = coerce_list_fields_to_tuples(cleaned_config)

    assert result == cleaned_config
