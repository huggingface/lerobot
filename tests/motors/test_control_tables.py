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

import importlib
from collections.abc import Iterator

import pytest

TABLES_MODULES = [
    "lerobot.motors.feetech.tables",
    "lerobot.motors.dynamixel.tables",
    "lerobot.motors.damiao.tables",
    "lerobot.motors.robstride.tables",
]


def _iter_string_keys(obj: object, path: str) -> Iterator[tuple[str, str]]:
    """Yield `(path, key)` for every string key of `obj`, descending into nested dicts."""
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        key_path = f"{path}[{key!r}]"
        if isinstance(key, str):
            yield key_path, key
        yield from _iter_string_keys(value, key_path)


@pytest.mark.parametrize("module_name", TABLES_MODULES)
def test_table_keys_have_no_surrounding_whitespace(module_name):
    """A register name such as `"Acceleration_Multiplier "` cannot be found by the name users type."""
    module = importlib.import_module(module_name)
    tables = {name: obj for name, obj in vars(module).items() if name.isupper() and isinstance(obj, dict)}
    assert tables, f"No table found in {module_name}"

    bad_keys = [
        key_path
        for name, table in tables.items()
        for key_path, key in _iter_string_keys(table, name)
        if key != key.strip()
    ]

    assert not bad_keys, f"Keys with leading or trailing whitespace in {module_name}: {bad_keys}"
