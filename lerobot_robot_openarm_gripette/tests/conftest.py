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

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np


class FakeGripperClient:
    """Stands in for ``gripette.client.GripperClient``: streams one 8x6 frame then idles until closed."""

    def __init__(self, addr: str):
        self.addr = addr
        self._closed = threading.Event()
        self.connect = MagicMock()
        self.torque_on = MagicMock()
        self.torque_off = MagicMock()
        self.move = MagicMock()
        self.close = MagicMock(side_effect=self._closed.set)

    def stream(self):
        ok, buf = cv2.imencode(".jpg", np.full((8, 6, 3), (200, 0, 0), dtype=np.uint8))
        assert ok
        yield SimpleNamespace(jpeg_data=buf.tobytes(), motor1=0.4, motor2=0.5)
        self._closed.wait()


# The real gripette package is a git-only dependency; tests never talk to a Gripette server.
sys.modules.setdefault("gripette", types.ModuleType("gripette"))
sys.modules.setdefault("gripette.client", types.ModuleType("gripette.client"))
sys.modules["gripette.client"].GripperClient = FakeGripperClient  # type: ignore[attr-defined]
