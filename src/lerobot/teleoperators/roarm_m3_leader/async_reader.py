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

"""Background read-only poller for the RoArm-M3 leader.

The leader arm is held torque-off and moved by hand. ``roarm_sdk``'s ``joints_angle_get``
blocks ~35-50 ms, so this thread polls continuously and caches the latest reading; the
synchronous teleop loop reads the cache (~0 ms) via ``get_pos()`` instead of blocking.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class AsyncArmReader:
    """Background thread that continuously polls ``joints_angle_get()`` and caches it."""

    def __init__(self, arm, name: str, sleep_between_reads_s: float = 0.005):
        self._arm = arm
        self._name = name
        self._sleep = sleep_between_reads_s
        self._pos: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"AsyncArmReader-{name}")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def get_pos(self) -> np.ndarray | None:
        with self._lock:
            return None if self._pos is None else self._pos.copy()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                pos = self._arm.joints_angle_get()
                if pos is not None:
                    arr = np.array(pos, dtype=np.float32)
                    with self._lock:
                        self._pos = arr
            except Exception as exc:  # pragma: no cover - serial flakiness
                logger.debug(f"AsyncArmReader [{self._name}] error: {exc}")
            time.sleep(self._sleep)
