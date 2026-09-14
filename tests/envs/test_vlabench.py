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

"""Check VLABench's declared action space against a pinned recorded action."""

import json
from pathlib import Path

import numpy as np

from lerobot.envs.vlabench import VLABenchEnv


def test_eef_action_space_contains_recorded_primitive_action():
    # Source: https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video/blob/9846a2f6bead3873251dc4fe3079359d57326b7c/data/chunk-000/file-000.parquet
    # First row (episode 0, frame 0); source SHA-256 is recorded in the fixture.
    fixture_path = Path(__file__).parents[1] / "artifacts" / "datasets" / "vlabench_primitive_action.json"
    fixture = json.loads(fixture_path.read_text())
    row = fixture["row"]
    action = np.asarray(row["actions"], dtype=np.float32)

    # Construction declares the real Box; the simulator is only created on reset().
    with VLABenchEnv() as env:
        assert env.action_space.contains(action), (
            f"Recorded primitive action (episode={row['episode_index']}, frame={row['frame_index']}) "
            f"{action.tolist()} is outside {env.action_space}"
        )
