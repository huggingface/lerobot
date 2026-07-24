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

import sys
from dataclasses import dataclass

import draccus
import pytest

from lerobot.configs.parser import wrap


@dataclass
class _Base(draccus.ChoiceRegistry):
    pass


@_Base.register_subclass("small")
@dataclass
class _SmallChoice(_Base):
    small_field: int = 1


@_Base.register_subclass("big")
@dataclass
class _BigChoice(_Base):
    big_field_a: int = 1
    big_field_b: int = 2
    big_field_c: int = 3


@dataclass
class _Config:
    choice: _Base | None = None


def test_help_scopes_to_selected_choice_type(capsys):
    @wrap()
    def dummy_func(cfg: _Config):
        return cfg

    sys.argv = ["dummy_script.py", "--choice.type=small", "--help"]

    with pytest.raises(SystemExit) as exc_info:
        dummy_func()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "small_field" in captured.out
    assert "big_field_a" not in captured.out
    assert "big_field_b" not in captured.out
    assert "big_field_c" not in captured.out
    # the full choice list must still be shown for validation purposes
    assert "small" in captured.out
    assert "big" in captured.out


def test_help_with_no_choice_selected_lists_choices_without_expanding(capsys):
    @wrap()
    def dummy_func(cfg: _Config):
        return cfg

    sys.argv = ["dummy_script.py", "--help"]

    with pytest.raises(SystemExit) as exc_info:
        dummy_func()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "small" in captured.out
    assert "big" in captured.out
    assert "small_field" not in captured.out
    assert "big_field_a" not in captured.out
