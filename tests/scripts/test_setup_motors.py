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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import lerobot.scripts.lerobot_setup_motors as motors_module


def test_main_registers_plugins_before_parsing(monkeypatch):
    calls = []
    monkeypatch.setattr(motors_module, "register_third_party_plugins", lambda: calls.append("register"))
    monkeypatch.setattr(motors_module, "setup_motors", lambda: calls.append("setup"))

    motors_module.main()

    assert calls == ["register", "setup"]


def test_setup_motors_accepts_third_party_device(monkeypatch):
    device = MagicMock()
    monkeypatch.setattr(motors_module, "make_teleoperator_from_config", lambda _: device)
    cfg = SimpleNamespace(device=SimpleNamespace(type="third_party"))

    motors_module.setup_motors.__wrapped__(cfg)

    device.setup_motors.assert_called_once_with()


def test_setup_motors_reports_unsupported_device(monkeypatch):
    device = object()
    monkeypatch.setattr(motors_module, "make_teleoperator_from_config", lambda _: device)
    cfg = SimpleNamespace(device=SimpleNamespace(type="third_party"))

    with pytest.raises(NotImplementedError, match="third_party"):
        motors_module.setup_motors.__wrapped__(cfg)
