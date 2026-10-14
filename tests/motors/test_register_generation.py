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

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate_motor_registers.py"
COMMITTED_OUTPUT = ROOT / "src/lerobot/motors/registers.py"


def run_generator(output: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output), *options],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_snapshot_matches_rustypot_definitions():
    rustypot = pytest.importorskip("rustypot")
    names = set()
    for controller in vars(rustypot).values():
        models = getattr(controller, "models", None)
        if isinstance(controller, type) and callable(models):
            for model in models():
                found = rustypot.find_model(model)
                assert found is not None
                definition = found[1]
                names.update(register.name for register in definition.registers())

    assert names
    expected = {spelling for name in names for spelling in (name.lower(), name.upper(), name.title())}
    committed = set(get_args(RegisterName.__value__))
    if version("rustypot") == "1.11.0":
        assert committed == expected
    else:
        # Later releases can add registers without changing the minimum-version snapshot.
        assert committed <= expected


def test_generator_writes_deterministic_output_and_checks_it(tmp_path):
    pytest.importorskip("rustypot")
    output = tmp_path / "registers.py"
    generated = run_generator(output)
    assert generated.returncode == 0, generated.stderr
    first = output.read_bytes()
    generated_again = run_generator(output)
    assert generated_again.returncode == 0, generated_again.stderr
    assert output.read_bytes() == first
    checked = run_generator(output, "--check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert output.read_bytes() == first
    if version("rustypot") == "1.11.0":
        assert first == COMMITTED_OUTPUT.read_bytes()


def test_check_reports_stale_and_missing_output_without_writes(tmp_path):
    pytest.importorskip("rustypot")
    output = tmp_path / "registers.py"
    stale = "# stale register types\n"
    output.write_text(stale)
    checked = run_generator(output, "--check")
    assert checked.returncode == 1, checked.stderr
    assert "-# stale register types" in checked.stdout
    assert "+type RegisterName = Literal[" in checked.stdout
    assert output.read_text() == stale

    output.unlink()
    missing = run_generator(output, "--check")
    assert missing.returncode == 1, missing.stderr
    assert "+type RegisterName = Literal[" in missing.stdout
    assert not output.exists()
