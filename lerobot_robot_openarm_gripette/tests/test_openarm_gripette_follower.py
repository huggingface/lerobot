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

import json

import numpy as np

from lerobot_robot_openarm_gripette.prediction_logger import PredictionLogger


def test_prediction_logger_writes_and_flushes_jsonl(tmp_path):
    logger = PredictionLogger(tmp_path)
    logger.write({"type": "metadata", "initial_joints_deg": np.array([1.0, 2.0])})
    logger.write(
        {
            "type": "prediction",
            "chunk_index": 0,
            "action_index": 1,
            "raw": {"x": np.float32(0.25), "closure": np.float64(0.8)},
            "joint_targets_deg": np.arange(7, dtype=np.float32),
            "executed": False,
        }
    )

    with logger.path.open() as f:
        records = [json.loads(line) for line in f]
    logger.close()

    assert records == [
        {"type": "metadata", "initial_joints_deg": [1.0, 2.0]},
        {
            "type": "prediction",
            "chunk_index": 0,
            "action_index": 1,
            "raw": {"x": 0.25, "closure": 0.8},
            "joint_targets_deg": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "executed": False,
        },
    ]
