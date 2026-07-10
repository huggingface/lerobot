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

from lerobot.utils.utils import flatten_dict, unflatten_dict


def test_flatten_simple():
    assert flatten_dict({"a": {"b": 1}, "c": 2}) == {"a/b": 1, "c": 2}


def test_unflatten_roundtrip():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    assert unflatten_dict(flatten_dict(nested)) == nested


def test_custom_sep():
    nested = {"x": {"y": 9}}
    assert flatten_dict(nested, sep=".") == {"x.y": 9}
    assert unflatten_dict({"x.y": 9}, sep=".") == nested


def test_empty_dict():
    assert flatten_dict({}) == {}
    assert unflatten_dict({}) == {}


def test_empty_nested_dict_preserved():
    # Empty nested dict is a value and must round-trip; previously flatten
    # dropped empty branches, which made stats-merge callsites silent.
    nested = {"stats": {}}
    flat = flatten_dict(nested)
    # Concrete contract: empty-dict leaves become terminal values in flat form.
    assert flat == {"stats": {}}
    assert unflatten_dict(flat) == nested
