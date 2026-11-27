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
"""Tests for the dependency-free feature helpers in ``lerobot.utils.feature_utils``."""

import pytest

from lerobot.utils.feature_utils import get_dimension_names


@pytest.mark.parametrize(
    "feature, expected",
    [
        pytest.param({"shape": (2,), "names": ["x", "y"]}, ["x", "y"], id="flat_list"),
        pytest.param({"shape": (2,), "names": ("x", "y")}, ["x", "y"], id="flat_tuple"),
        pytest.param({"names": ["x", "y"]}, ["x", "y"], id="flat_list_without_shape"),
        pytest.param({"shape": (2,), "names": {"motors": ["m0", "m1"]}}, ["m0", "m1"], id="one_group"),
        pytest.param(
            {"shape": (3,), "names": {"arm": ["m0", "m1"], "gripper": ["g"]}},
            ["m0", "m1", "g"],
            id="several_groups",
        ),
        pytest.param(
            {"shape": (2,), "names": {"delta_x": 0, "delta_y": 1}}, ["delta_x", "delta_y"], id="name_to_index"
        ),
    ],
)
def test_get_dimension_names_flattens_every_layout(feature, expected):
    assert get_dimension_names(feature) == expected


@pytest.mark.parametrize(
    "feature",
    [
        pytest.param(None, id="no_feature"),
        pytest.param({}, id="empty_feature"),
        pytest.param({"shape": (2,)}, id="missing_names"),
        pytest.param({"shape": (2,), "names": None}, id="none_names"),
        pytest.param({"shape": (0,), "names": []}, id="empty_names"),
        # bool is a subclass of int, but True/False are not dimension indices.
        pytest.param({"shape": (2,), "names": {"a": True, "b": False}}, id="bool_values"),
        pytest.param({"shape": (2,), "names": {"a": 0, "b": 2}}, id="index_gap"),
        # Frames are recorded in key order, so indices that disagree with it are ambiguous.
        pytest.param({"shape": (2,), "names": {"b": 1, "a": 0}}, id="index_out_of_key_order"),
        pytest.param({"shape": (3,), "names": ["x", "y"]}, id="fewer_names_than_shape"),
        pytest.param({"shape": (2,), "names": {"motors": ["m0", "m1", "m2"]}}, id="group_longer_than_shape"),
        pytest.param({"shape": (2,), "names": [0, 1]}, id="non_string_names"),
        # An image's names label its axes, not its dimensions.
        pytest.param({"shape": (64, 96, 3), "names": ["height", "width", "channels"]}, id="image"),
        pytest.param({"shape": (2, 2), "names": ["a", "b"]}, id="2d"),
    ],
)
def test_get_dimension_names_returns_none_without_usable_names(feature):
    assert get_dimension_names(feature) is None


def test_get_dimension_names_returns_a_copy():
    names = ["x", "y"]
    result = get_dimension_names({"shape": (2,), "names": names})
    assert result == names
    assert result is not names
