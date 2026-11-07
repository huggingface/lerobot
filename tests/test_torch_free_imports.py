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
"""Modules that must import without torch.

Each module is imported in a fresh interpreter where torch is installed but refuses to load, so a
failure's traceback points at the import that pulled torch in.
"""

import subprocess
import sys

import pytest

from lerobot.utils.import_utils import _datasets_available

# The spec is still found, so code that only checks whether torch is installed keeps working. Each
# load attempt is recorded, so an import wrapped in try/except still fails the test.
BLOCK_TORCH = """
import importlib.abc
import importlib.machinery
import sys
import traceback

attempts = []


class RefuseLoader(importlib.abc.Loader):
    def exec_module(self, module):
        attempts.append("".join(traceback.format_stack()))
        raise RuntimeError("torch must not be imported here")


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = RefuseLoader()
        return spec


sys.meta_path.insert(0, BlockTorch())
"""

IMPORT_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
importlib.import_module(sys.argv[1])
if attempts:
    sys.exit(attempts[0])
"""
)


@pytest.mark.parametrize(
    "module",
    [
        "lerobot.inference",
        "lerobot.scripts.lerobot_calibrate",
        "lerobot.scripts.lerobot_teleoperate",
    ],
)
def test_imports_without_torch(module):
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITHOUT_TORCH, module], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.dataset
@pytest.mark.skipif(not _datasets_available, reason="datasets not installed")
def test_rollout_parses_and_rejects_a_bad_command_line_without_torch():
    code = (
        BLOCK_TORCH
        + """
from draccus.utils import ParsingError

import lerobot.scripts.lerobot_rollout

sys.argv = ["lerobot-rollout", "--strategy.type=base"]
try:
    lerobot.scripts.lerobot_rollout.main()
except ParsingError as e:
    print("REJECTED", e.__cause__)
if attempts:
    sys.exit(attempts[0])
"""
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "REJECTED --robot.type is required for rollout" in result.stdout
