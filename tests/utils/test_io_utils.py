#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lerobot.utils.io_utils import deserialize_json_into_object


@pytest.fixture
def tmp_json_file(tmp_path: Path):
    """Writes `data` to a temporary JSON file and returns the file's path."""

    def _write(data: Any) -> Path:
        file_path = tmp_path / "data.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        return file_path

    return _write


def test_simple_dict(tmp_json_file):
    data = {"name": "Alice", "age": 30}
    json_path = tmp_json_file(data)
    obj = {"name": "", "age": 0}
    assert deserialize_json_into_object(json_path, obj) == data


def test_nested_structure(tmp_json_file):
    data = {"items": [1, 2, 3], "info": {"active": True}}
    json_path = tmp_json_file(data)
    obj = {"items": [0, 0, 0], "info": {"active": False}}
    assert deserialize_json_into_object(json_path, obj) == data


def test_tuple_conversion(tmp_json_file):
    data = {"coords": [10.5, 20.5]}
    json_path = tmp_json_file(data)
    obj = {"coords": (0.0, 0.0)}
    result = deserialize_json_into_object(json_path, obj)
    assert result["coords"] == (10.5, 20.5)


def test_type_mismatch_raises(tmp_json_file):
    data = {"numbers": {"bad": "structure"}}
    json_path = tmp_json_file(data)
    obj = {"numbers": [0, 0]}
    with pytest.raises(TypeError):
        deserialize_json_into_object(json_path, obj)


def test_missing_key_raises(tmp_json_file):
    data = {"one": 1}
    json_path = tmp_json_file(data)
    obj = {"one": 0, "two": 0}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


def test_extra_key_raises(tmp_json_file):
    data = {"one": 1, "two": 2}
    json_path = tmp_json_file(data)
    obj = {"one": 0}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


def test_list_length_mismatch_raises(tmp_json_file):
    data = {"nums": [1, 2, 3]}
    json_path = tmp_json_file(data)
    obj = {"nums": [0, 0]}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


@pytest.mark.parametrize(
    "obj,data,error",
    [
        ({"nested": {"step": 0}, "late": 0}, {"nested": {"step": 3}, "late": "bad"}, TypeError),
        ([0, {"rate": 0.0}], [5, {"rate": "bad"}], TypeError),
        (([0], {"step": 0}), [[5], {"step": "bad"}], TypeError),
        ({"values": [0, 0], "late": []}, {"values": [1, 2], "late": [1]}, ValueError),
        ({"values": [0, 0], "late": {"x": 0}}, {"values": [1, 2], "late": {"y": 1}}, ValueError),
    ],
    ids=["nested-dict", "nested-list", "mutable-in-tuple", "late-length", "late-keys"],
)
def test_invalid_restore_preserves_entire_object(tmp_json_file, obj, data, error):
    original = copy.deepcopy(obj)

    with pytest.raises(error):
        deserialize_json_into_object(tmp_json_file(data), obj)

    assert obj == original


def test_valid_restore_preserves_mutable_container_identity(tmp_json_file):
    values = [0, 0]
    nested = {"values": values}
    obj = {"nested": nested, "coords": (values, 0)}
    data = {"nested": {"values": [1, 2]}, "coords": [[1, 2], 3]}

    result = deserialize_json_into_object(tmp_json_file(data), obj)

    assert result is obj
    assert result["nested"] is nested
    assert result["nested"]["values"] is values
    assert result["coords"][0] is values
    assert result == {"nested": {"values": [1, 2]}, "coords": ([1, 2], 3)}
