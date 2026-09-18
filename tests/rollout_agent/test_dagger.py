# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import numpy as np
import pytest

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.language import language_feature_info
from lerobot.rollout.agent.dagger import build_dataset


def make_dataset(root, repo, correction=False):
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        **language_feature_info(),
    }
    if correction:
        features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
    ds = LeRobotDataset.create(
        repo, root=root, fps=10, robot_type="fake", features=features, use_videos=False
    )
    for i in range(5):
        frame = {
            "action": np.array([i], dtype=np.float32),
            "observation.state": np.array([0], dtype=np.float32),
            "task": "pick cup",
            "language_persistent": [
                {
                    "role": "assistant",
                    "style": "subtask",
                    "content": "grasp cup",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                }
            ],
            "language_events": [],
        }
        if correction:
            frame["intervention"] = np.array([i in (1, 2, 4)], dtype=bool)
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()


def test_aggregation_uses_only_selected_executed_intervention_spans(tmp_path):
    make_dataset(tmp_path / "seed", "local/seed")
    make_dataset(tmp_path / "correction", "local/correction", correction=True)
    spec = {
        "seed": {"repo_id": "local/seed", "root": str(tmp_path / "seed"), "episodes": [0]},
        "corrections": [
            {"repo_id": "local/correction", "root": str(tmp_path / "correction"), "episodes": [0]}
        ],
        "output_repo_id": "local/merged",
        "output_root": str(tmp_path / "merged"),
    }
    result = build_dataset(spec)
    assert result["episodes"] == 3  # Seed episode and two disjoint correction spans.
    output = LeRobotDataset("local/merged", root=tmp_path / "merged")
    assert len(output) == 8
    assert [float(output[i]["action"]) for i in range(5, 8)] == [1.0, 2.0, 4.0]
    assert float(output[5]["timestamp"]) == 0
    assert output[5]["language_persistent"][0]["timestamp"] == 0
    assert len(LeRobotDataset("local/correction", root=tmp_path / "correction")) == 5
    with pytest.raises(ValueError, match="new dataset"):
        build_dataset(spec)
