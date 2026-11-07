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


def test_robot_processors_run_and_load_without_torch(tmp_path):
    code = (
        RECORD_TORCH
        + f"""
import numpy as np
from lerobot.processor import (
    RenameObservationsProcessorStep,
    RobotProcessorPipeline,
    make_default_processors,
    observation_to_transition,
    transition_to_observation,
)

teleop_action, robot_action, robot_observation = make_default_processors()
action = {{"shoulder_pan.pos": 1.0, "gripper.pos": 0.5}}
observation = {{"shoulder_pan.pos": 0.9, "front": np.zeros((4, 4, 3), np.uint8)}}
assert robot_action((teleop_action((action, observation)), observation)) == action
assert robot_observation(observation).keys() == observation.keys()
robot_action.save_pretrained({str(tmp_path)!r})

# A registered step is saved by its name, so loading it looks the name up.
converters = {{"to_transition": observation_to_transition, "to_output": transition_to_observation}}
renamer = RobotProcessorPipeline(steps=[RenameObservationsProcessorStep(rename_map={{"front": "wrist"}})], **converters)
renamer.save_pretrained({str(tmp_path)!r}, config_filename="renamer.json")
loaded = RobotProcessorPipeline.from_pretrained({str(tmp_path)!r}, config_filename="renamer.json", **converters)
assert loaded(observation).keys() == {{"shoulder_pan.pos", "wrist"}}
"""
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
