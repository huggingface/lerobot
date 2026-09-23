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

"""Robot-side processing without torch, checked in fresh interpreters.

Test modules import torch as they are collected, so each case runs in a new Python process.
"""

import subprocess
import sys

from tests.test_torch_free_imports import BLOCK_TORCH

# Exits with an error at the end if anything tried to load torch, even inside try/except.
RECORD_TORCH = (
    BLOCK_TORCH
    + """
import atexit, os
atexit.register(lambda: attempts and (print(attempts[0], file=sys.stderr), os._exit(3)))
"""
)


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)


def test_default_robot_processors_run_without_torch(tmp_path):
    result = run(
        RECORD_TORCH
        + f"""
import numpy as np
from lerobot.processor import make_default_processors

teleop_action, robot_action, robot_observation = make_default_processors()
action = {{"shoulder_pan.pos": 1.0, "gripper.pos": 0.5}}
observation = {{"shoulder_pan.pos": 0.9, "front": np.zeros((4, 4, 3), np.uint8)}}
assert robot_action((teleop_action((action, observation)), observation)) == action
assert robot_observation(observation).keys() == observation.keys()
robot_action.save_pretrained({str(tmp_path)!r})
"""
    )
    assert result.returncode == 0, result.stderr


def test_saved_pipeline_with_a_processor_step_loads(tmp_path):
    save = (
        "from lerobot.processor import DataProcessorPipeline, DeviceProcessorStep\n"
        f"DataProcessorPipeline([DeviceProcessorStep(device='cpu')]).save_pretrained({str(tmp_path)!r})\n"
    )
    load = (
        "import sys\n"
        "from lerobot.processor.pipeline import DataProcessorPipeline\n"
        f"p = DataProcessorPipeline.from_pretrained({str(tmp_path)!r}, config_filename='dataprocessorpipeline.json')\n"
        "print('STEPS', [type(s).__name__ for s in p.steps])\n"
        "print('POLICIES', sorted({m.split('.')[2] for m in sys.modules if m.startswith('lerobot.policies.')}))\n"
    )
    for code in (save, load):
        result = run(code)
        assert result.returncode == 0, result.stderr
    assert "STEPS ['DeviceProcessorStep']" in result.stdout
    # A step defined in lerobot.processor must not make the registry import every policy.
    assert "POLICIES []" in result.stdout


def test_unknown_step_name_raises_key_error_without_torch():
    result = run(
        RECORD_TORCH
        + """
from lerobot.processor import ProcessorStepRegistry

try:
    ProcessorStepRegistry.get("no_such_step")
except KeyError as e:
    assert "device_processor" in str(e), e
else:
    raise AssertionError("no KeyError")
"""
    )
    assert result.returncode == 0, result.stderr
