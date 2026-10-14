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

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import get_args

import pytest

from lerobot.motors.registers import RegisterName


def test_register_names_match_servo_definitions():
    rustypot = pytest.importorskip("rustypot")
    names = set()
    for controller in vars(rustypot).values():
        models = getattr(controller, "models", None)
        if isinstance(controller, type) and callable(models):
            for model in models():
                found = rustypot.find_model(model)
                assert found is not None
                names.update(register.name for register in found[1].registers())
    assert names
    expected = {value for name in names for value in (name, name.upper(), name.title())}
    actual = set(get_args(RegisterName.__value__))
    assert actual <= expected  # Newer rustypot releases can add names.
    if version("rustypot") == "1.11.0":
        assert actual == expected


def test_generated_source_matches_snapshot():
    pytest.importorskip("rustypot")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/generate_motor_registers.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout
    if version("rustypot") == "1.11.0":
        assert result.stdout == (root / "src/lerobot/motors/registers.py").read_text()
