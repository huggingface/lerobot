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
"""`check_input` raises in eager mode and stays a runtime assert in a saved or decomposed exported program."""

import io
import subprocess
import sys

import pytest
import torch

from lerobot.policies.common.input_checks import check_input


class Positive(torch.nn.Module):
    def forward(self, x):
        check_input(x > 0, "x must be positive")
        return x * 2


GOOD, BAD = torch.ones(4), torch.tensor([1.0, -1.0, 1.0, 1.0])


def test_eager_check_raises_its_error_type_and_message():
    torch.testing.assert_close(Positive()(GOOD), GOOD * 2)
    with pytest.raises(ValueError, match="x must be positive"):
        Positive()(BAD)
    with pytest.raises(AssertionError, match="built on failure"):
        check_input(False, lambda: "built on failure", error_type=AssertionError)


# TRANSFORMERS_DISABLE_TORCH_CHECK=1 turns off transformers' own checks; it must not remove these asserts.
@pytest.mark.parametrize("transformers_checks_off", [False, True], ids=["switch-unset", "switch-set"])
def test_exported_assert_survives_saving_and_decomposition(monkeypatch, transformers_checks_off):
    if transformers_checks_off:
        monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    program = torch.export.export(Positive(), (GOOD,), strict=False)
    buffer = io.BytesIO()
    torch.export.save(program, buffer)
    buffer.seek(0)
    for lowered in (program, torch.export.load(buffer), program.run_decompositions()):
        torch.testing.assert_close(lowered.module()(GOOD), GOOD * 2)
        with pytest.raises(RuntimeError, match="Runtime assertion failed"):
            lowered.module()(BAD)


def test_compiled_check_raises_with_transformers_checks_off(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    compiled = torch.compile(Positive(), backend="eager")
    torch.testing.assert_close(compiled(GOOD), GOOD * 2)
    with pytest.raises(ValueError, match="x must be positive"):
        compiled(BAD)


def test_export_does_not_need_transformers():
    code = """
import sys
# As if transformers, and peft which needs it, were not installed.
sys.modules["transformers"] = sys.modules["peft"] = None
import torch
from lerobot.policies.common.input_checks import check_input

class Positive(torch.nn.Module):
    def forward(self, x):
        check_input(x > 0, "x must be positive")
        return x * 2

program = torch.export.export(Positive(), (torch.ones(2),), strict=False).module()
try:
    program(torch.tensor([1.0, -1.0]))
except RuntimeError:
    print("refused")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("refused"), result.stdout
