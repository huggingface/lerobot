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

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from lerobot.motors import CustomRegister
from lerobot.motors.motors_bus import Motor, MotorNormMode
from tests.mocks.mock_motors_bus import MockMotorsBus

REPO_ROOT = Path(__file__).parents[2]
REGISTER_MODULE = REPO_ROOT / "src" / "lerobot" / "motors" / "registers.py"
GENERATOR = REPO_ROOT / "scripts" / "generate_motor_registers.py"

# One call per checked method, on names that are valid but misspelled, plus the valid
# spellings and the `CustomRegister` escape the static check must keep accepting.
SNIPPET = """\
from lerobot.motors import CustomRegister, SerialMotorsBus


def probe(bus: SerialMotorsBus, name: str) -> None:
    bus.read("Present_Position", "joint")
    bus.read("present_position", "joint")
    bus.read("PRESENT_POSITION", "joint")
    bus.write("Goal_Position", "joint", 1)
    bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", {"joint": 1})
    bus.read(CustomRegister(name), "joint")
    bus.write("Present_Positon", "joint", 1)
    bus.sync_read("Present_Positionn")
    bus.sync_write("Goal_Pozition", {"joint": 1})
"""

TYPOS = ("Present_Positon", "Present_Positionn", "Goal_Pozition")


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_motor_registers", GENERATOR)
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_members() -> list[str]:
    """The string members of the generated `RegisterName` literal, in file order."""
    tree = ast.parse(REGISTER_MODULE.read_text())
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "RegisterName" for target in node.targets)
        ):
            continue
        literal = node.value
        assert isinstance(literal, ast.Subscript) and isinstance(literal.slice, ast.Tuple)
        return [constant.value for constant in literal.slice.elts]
    raise AssertionError("RegisterName not found in the generated module.")


@pytest.fixture(scope="session")
def mypy_output(tmp_path_factory) -> str:
    """mypy's report on the snippet, computed once for the whole test session."""
    mypy_api = pytest.importorskip("mypy.api")
    cache = tmp_path_factory.mktemp("mypy-cache")
    snippet = tmp_path_factory.mktemp("snippet") / "snippet.py"
    snippet.write_text(SNIPPET)
    # Point mypy at the source tree: an editable install is invisible to it, so without
    # this the snippet's imports degrade to `Any` and nothing is checked.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("MYPYPATH", str(REPO_ROOT / "src"))
        report, errors, _ = mypy_api.run(
            [
                str(snippet),
                "--cache-dir",
                str(cache),
                "--python-executable",
                sys.executable,
                # The repository config skips followed imports, which would erase the types
                # the check needs; the snippet wants the default full analysis.
                "--follow-imports=normal",
            ]
        )
    assert not errors, errors
    return report


def test_mypy_rejects_a_register_typo_on_every_method(mypy_output):
    """The static check is the point of the generated literal: a misspelled register is an
    arg-type error on each of the four methods, while valid spellings and `CustomRegister`
    pass."""
    flagged = [line for line in mypy_output.splitlines() if "arg-type" in line]

    for typo in TYPOS:
        assert any(typo in line for line in flagged), f"'{typo}' not flagged:\n{mypy_output}"
    assert len(flagged) == len(TYPOS), f"unexpected additional errors:\n{mypy_output}"


def test_generated_literal_is_consistent():
    """Three spellings of one register per group, all lowering to the same lowercase name."""
    members = literal_members()

    assert len(members) % 3 == 0
    lowers = []
    for group in [members[i : i + 3] for i in range(0, len(members), 3)]:
        assert {spelling.lower() for spelling in group} == {group[0].lower()}
        assert len(set(group)) == 3, f"not three distinct spellings: {group}"
        lowers.append(group[0].lower())
    assert len(lowers) == len(set(lowers)), "a register is spelled twice."
    assert "present_position" in lowers


def test_generated_literal_covers_rustypot():
    """Every register rustypot ships is in the generated literal (the reverse is allowed: a
    newer rustypot may add registers until the file is regenerated)."""
    pytest.importorskip("rustypot")
    rustypot_names = set(load_generator().collect_register_names())

    missing = rustypot_names - set(literal_members())
    assert not missing, f"registers rustypot has that the literal lacks: {sorted(missing)}"


def test_custom_register_reaches_the_bus_at_run_time():
    """The escape hatch is an ordinary string at run time: the bus lowers it like any other."""
    bus = MockMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    bus._bus.seed(1, "model_number", 5678)

    assert bus.read(CustomRegister("MODEL_NUMBER"), "dummy") == 5678
    assert bus._bus.reads == [(1, "model_number")]
