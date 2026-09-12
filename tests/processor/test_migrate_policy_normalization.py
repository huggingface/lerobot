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


def test_coerce_list_fields_to_tuples_converts_tuple_typed_fields():
    # DiffusionConfig declares crop_shape: tuple[int, int] | None and down_dims: tuple[int, ...]
    cleaned_config = {"crop_shape": [84, 84], "down_dims": [512, 1024, 2048]}

    result = coerce_list_fields_to_tuples(cleaned_config, "diffusion")

    assert result["crop_shape"] == (84, 84)
    assert result["down_dims"] == (512, 1024, 2048)


def test_coerce_list_fields_to_tuples_leaves_genuine_list_fields_as_lists():
    # PI0Config declares relative_exclude_joints: list[str], not a tuple. Coercing it would
    # silently break any downstream code that mutates the list (the bug a reviewer caught).
    cleaned_config = {"relative_exclude_joints": ["gripper"]}

    result = coerce_list_fields_to_tuples(cleaned_config, "pi0")

    assert result["relative_exclude_joints"] == ["gripper"]
    assert isinstance(result["relative_exclude_joints"], list)


def test_coerce_list_fields_to_tuples_leaves_molmoact2_image_keys_as_a_list():
    # Same shape of bug as above, on a different policy's top-level list field.
    cleaned_config = {"image_keys": ["front", "wrist"]}

    result = coerce_list_fields_to_tuples(cleaned_config, "molmoact2")

    assert result["image_keys"] == ["front", "wrist"]
    assert isinstance(result["image_keys"], list)


def test_coerce_list_fields_to_tuples_skips_feature_dicts():
    # input_features/output_features are dicts of PolicyFeature by the time this runs, not
    # lists, so they're left alone regardless of their declared type.
    features = {"observation.image": object()}
    cleaned_config = {"input_features": features, "output_features": features}

    result = coerce_list_fields_to_tuples(cleaned_config, "diffusion")

    assert result["input_features"] is features
    assert result["output_features"] is features


def test_coerce_list_fields_to_tuples_leaves_non_list_values_untouched():
    cleaned_config = {"horizon": 16, "vision_backbone": "resnet18", "crop_shape": None}

    result = coerce_list_fields_to_tuples(cleaned_config, "diffusion")

    assert result == cleaned_config


def test_coerce_list_fields_to_tuples_ignores_unrelated_keys():
    # A key that isn't a field on the target config at all (e.g. leftover from an older
    # config format) shouldn't crash the lookup; it's just left as-is.
    cleaned_config = {"some_removed_field": [1, 2, 3]}

    result = coerce_list_fields_to_tuples(cleaned_config, "diffusion")

    assert result["some_removed_field"] == [1, 2, 3]
