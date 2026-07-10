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

from lerobot.utils.utils import TimerManager


def test_stop_without_start_is_safe():
    timer = TimerManager(log=False)
    assert timer.stop() == 0.0
    assert timer.count == 0


def test_double_stop_is_safe():
    timer = TimerManager(log=False)
    timer.start()
    time.sleep(0.001)
    first = timer.stop()
    assert first > 0.0
    assert timer.stop() == 0.0
    assert timer.count == 1


def test_context_manager_records():
    timer = TimerManager(log=False)
    with timer:
        time.sleep(0.001)
    assert timer.count == 1
    assert timer.last > 0.0
