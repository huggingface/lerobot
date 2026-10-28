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

import importlib.machinery
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


class FakeGraspProjection:
    def encode(self, proximal: float, distal: float) -> tuple[float, float]:
        return distal / 3.0, proximal / 2.0

    def decode(self, strategy: float, closure: float) -> tuple[float, float]:
        return closure * 2.0, strategy * 3.0


def _fake_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    # find_spec() raises if a sys.modules entry has no spec. LeRobot probes `gripette` at import.
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


# The real gripette package is a git-only dependency; tests never talk to a Gripette server.
sys.modules.setdefault("gripette", _fake_module("gripette"))
sys.modules.setdefault("gripette.client", _fake_module("gripette.client"))
sys.modules.setdefault("gripette.grasp_projection", _fake_module("gripette.grasp_projection"))
sys.modules["gripette.client"].GripperClient = FakeGripperClient  # type: ignore[attr-defined]
sys.modules["gripette.grasp_projection"].GraspProjection = FakeGraspProjection  # type: ignore[attr-defined]
