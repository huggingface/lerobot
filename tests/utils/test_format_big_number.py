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

import importlib.util
from pathlib import Path

import pytest

# Load utils module uniquely to avoid lerobot package __init__ heavy deps in thin CI envs.
_UTILS_PATH = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "utils" / "utils.py"


def _load_format_big_number():
    # Fall back to package import when available; else load campus file stubs for accelerate typing
    try:
        from lerobot.utils.utils import format_big_number

        return format_big_number
    except Exception:
        pass
    # Minimal stub path: only the pure function is needed — exec just that def after they import less?
    # Prefer package; skip load if deps missing.
    pytest.importorskip("draccus")
    from lerobot.utils.utils import format_big_number

    return format_big_number


@pytest.fixture(scope="module")
def format_big_number():
    return _load_format_big_number()


def test_format_small_integers(format_big_number):
    assert format_big_number(999) == "999"
    assert format_big_number(1000) == "1K"
    assert format_big_number(1_500_000) == "2M"  # precision 0


def test_format_precision(format_big_number):
    assert format_big_number(1500, precision=1) == "1.5K"


def test_format_beyond_quintillion_is_string_with_q(format_big_number):
    # 1e18 -> 1Q after all divisions; 1e21 would leave residual under Q
    huge = 10**21  # 1e21 → remain 1re73 under Q? 1e21/1e3^5 = 1e21/1e15 = 1e6 → "1000000Q"
    out = format_big_number(huge, precision=0)
    assert isinstance(out, str)
    assert out.endswith("Q")
    assert "none" not in out.lower()


def test_format_negative(format_big_number):
    assert format_big_number(-2500, precision=1) == "-2.5K"
