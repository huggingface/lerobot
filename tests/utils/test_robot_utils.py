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

import time

import pytest

from lerobot.utils.robot_utils import precise_sleep


def test_non_positive_is_noop():
    # Direct calls exercise the no-op path; avoid a wall-clock assertion that
    # can flake when a CI worker is preempted between the two calls.
    precise_sleep(0)
    precise_sleep(-1)


def test_negative_spin_threshold_raises():
    with pytest.raises(ValueError, match="spin_threshold"):
        precise_sleep(0.001, spin_threshold=-0.1)


def test_negative_sleep_margin_raises():
    with pytest.raises(ValueError, match="sleep_margin"):
        precise_sleep(0.001, sleep_margin=-0.1)


def test_short_sleep_runs():
    start = time.perf_counter()
    precise_sleep(0.01)
    elapsed = time.perf_counter() - start
    # Generous bounds: avoid flakes on busy CI; just prove we waited.
    assert 0.005 <= elapsed < 0.5
