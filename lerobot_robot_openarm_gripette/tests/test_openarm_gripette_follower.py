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
from unittest.mock import MagicMock

import numpy as np

from lerobot_robot_openarm_gripette.manual_start import prepare_manual_start
from lerobot_robot_openarm_gripette.prediction_logger import PredictionLogger


def test_prepare_manual_start_disables_torque_captures_pose_and_holds():
    robot = MagicMock()
    events = []
    robot.bus.disable_torque.side_effect = lambda: events.append("disable")
    robot.bus.enable_torque.side_effect = lambda: events.append("enable")
    robot.bus.configure_motors.side_effect = lambda: events.append("configure")
    robot.send_action.side_effect = lambda action, **_: events.append(
        "open_gripper" if "proximal.pos" in action else "hold"
    )
    robot.get_observation.return_value = {f"joint_{i}.pos": float(i) for i in range(1, 8)}
    robot.config.position_kp = [100.0] * 7
    robot.config.position_kd = [2.0] * 7

    result = prepare_manual_start(
        robot,
        np.array([1, -1, 1, -1, 1, -1, 1]),
        gripper_action={"proximal.pos": 1.0, "distal.pos": 1.0},
        input_fn=lambda _: events.append("prompt"),
        sleep_fn=lambda _: events.append("sleep"),
        engage_duration_s=0.1,
        engage_frequency_hz=20.0,
    )

    assert events == [
        "disable",
        "sleep",
        "disable",
        "open_gripper",
        "prompt",
        "disable",
        "configure",
        "hold",
        "enable",
        "hold",
        "sleep",
        "hold",
        "sleep",
    ]
    assert result.tolist() == [1, -2, 3, -4, 5, -6, 7]
    gripper, first, _, last = robot.send_action.call_args_list
    assert gripper.args == ({"proximal.pos": 1.0, "distal.pos": 1.0},)
    assert set(first.kwargs["custom_kp"].values()) == {0.0}
    assert set(last.kwargs["custom_kp"].values()) == {100.0}


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
