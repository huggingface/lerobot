# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.steering_commands import SteeringCommands


def manifest():
    return {
        "version": 1,
        "source": {"repo_id": "test/data", "revision": "abc"},
        "segments": [
            {
                "episode_index": 3,
                "start_frame": 0,
                "end_frame": 5,
                "review": {"verdict": "accepted", "reviewer": "test-reviewer"},
                "commands": [
                    {"style": "subtask", "text": "reach for the tape", "evidence": "test video"},
                    {"style": "motion", "text": "move the left gripper left", "evidence": "test FK"},
                ],
            }
        ],
    }


def test_fresh_commands_preserve_action_targets_and_requested_mixture():
    index = SteeringCommands(manifest())
    sample = {
        "index": 400,
        "episode_index": 3,
        "frame_index": 2,
        "task": "put tape in bin",
        "action": object(),
    }
    np.random.seed(123)
    results = [index.sample(sample, 0.2) for _ in range(2000)]
    assert all(r["action"] is sample["action"] for r in results)
    assert len({r["task"] for r in results}) == 3
    assert 330 < sum(r["task"] == sample["task"] for r in results) < 470
    assert index.sample(sample, 0.2, deterministic=True) == index.sample(sample, 0.2, deterministic=True)


def test_missing_and_unreviewed_commands_fail_instead_of_changing_the_mixture():
    data = manifest()
    data["segments"][0]["review"]["verdict"] = "uncertain"
    with pytest.raises(ValueError, match="accepted"):
        SteeringCommands(data)
    index = SteeringCommands(manifest())
    with pytest.raises(ValueError, match="Missing"):
        index.at(3, 5)
    with pytest.raises(ValueError, match="Missing"):
        index.at(4, 0)
    data = manifest()
    data["segments"].append(copy.deepcopy(data["segments"][0]))
    with pytest.raises(ValueError, match="Overlapping"):
        SteeringCommands(data)


def test_camera_scoped_points_keep_original_coordinate_frame():
    data = manifest()
    command = {
        "style": "point",
        "text": "reach with the left gripper to",
        "evidence": "test tracker",
        "camera": "observation.images.base",
        "image_size": [640, 480],
        "points": [[100, 200]],
    }
    data["segments"][0]["commands"] = [command]
    result = SteeringCommands(data).sample({"episode_index": 3, "frame_index": 2, "task": "task"}, 0)
    assert result["task"] == "In base view (640x480 pixels), reach with the left gripper to: [100, 200]."
    command["points"] = [[640, 200]]
    with pytest.raises(ValueError, match="outside"):
        SteeringCommands(data)


def test_fk_uses_calibrated_measured_joints_and_rejects_reversals(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[2] / "examples/rebot_agent/fk_motion.py"))
    urdf = tmp_path / "test.urdf"
    urdf.write_text("test fixture, not a robot model")
    config = {
        "arm": "left",
        "units": "radians",
        "calibration": "test calibration",
        "frame": "left_base",
        "urdf": str(urdf),
        "joint_names": ["joint"],
        "state_keys": ["left.joint"],
        "signs": [-1],
        "offset_degrees": [10],
        "tool_frame": "tcp",
        "deadband_m": 0.01,
        "axis_directions": [{"positive": "right", "negative": "left"}] * 3,
    }

    class Kinematics:
        def forward_kinematics(self, joints):
            result = np.eye(4)
            result[0, 3] = joints[0] / 100
            return result

    extract = module["extract_motion"]
    motion = extract([[0], [np.pi / 2]], ["left.joint"], config, kinematics=Kinematics())
    assert motion[0]["text"] == "move the left gripper left"
    assert motion[0]["evidence"]["displacement_m"][0] == pytest.approx(-0.9)
    with pytest.raises(ValueError, match="reversing"):
        extract([[0], [np.pi / 2], [0.1]], ["left.joint"], config, kinematics=Kinematics())
