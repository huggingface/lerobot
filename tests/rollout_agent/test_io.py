# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from lerobot.rollout.agent.provider import public_snapshot
from lerobot.rollout.agent.robot_io import RealRobotIO


def test_chunk_postprocessing_uses_original_observation_once():
    robot_io = RealRobotIO.__new__(RealRobotIO)
    features = {
        "action": {"dtype": "float32", "names": ["joint.pos"], "shape": (1,)},
        "observation.state": {"dtype": "float32", "names": ["joint.pos"], "shape": (1,)},
    }
    pre = MagicMock(side_effect=lambda observation: observation)
    post = MagicMock(side_effect=lambda action: action + 2)
    policy = MagicMock()
    policy.predict_action_chunk.return_value = torch.tensor([[[1.0], [3.0]]])
    robot_io.ctx = SimpleNamespace(
        policy=SimpleNamespace(policy=policy, preprocessor=pre, postprocessor=post),
        data=SimpleNamespace(dataset_features=features),
        runtime=SimpleNamespace(cfg=SimpleNamespace(device="cpu")),
    )
    robot_io.robot = SimpleNamespace(robot_type="test")
    robot_io._revision = None
    snapshot = {"revision": 1, "task": "pick", "raw": {"_processed": {"joint.pos": 2.0}}}
    assert robot_io.predict(snapshot) == [{"joint.pos": 3.0}, {"joint.pos": 5.0}]
    assert post.call_count == 1
    policy.reset.assert_called_once()
    robot_io.predict(snapshot)
    policy.reset.assert_called_once()
    robot_io.predict({**snapshot, "revision": 2})
    assert policy.reset.call_count == 2


def test_record_actual_actions_and_language_after_clipping():
    robot_io = RealRobotIO.__new__(RealRobotIO)
    robot_io.ctx = SimpleNamespace(
        data=SimpleNamespace(
            dataset_features={
                "action": {"dtype": "float32", "names": ["joint.pos"], "shape": (1,)},
                "observation.state": {"dtype": "float32", "names": ["joint.pos"], "shape": (1,)},
            }
        )
    )
    robot_io.dataset = MagicMock()
    robot_io.journal = io.StringIO()
    robot_io.frames = robot_io.intervention_frames = robot_io.policy_frames = 0
    robot_io.fps = 30
    snapshot = {
        "task": "put all objects in bin",
        "raw": {"_processed": {"joint.pos": 0.2}},
        "id": 3,
        "observed_at": 1.0,
    }
    robot_io.record(snapshot, {"joint.pos": 7.0}, {"joint.pos": 0.5}, "tool", "grasp tape")
    frame = robot_io.dataset.add_frame.call_args.args[0]
    assert frame["action"].tolist() == [0.5]
    assert frame["intervention"].tolist() == [True]
    assert frame["language_persistent"][0]["content"] == "grasp tape"
    event = json.loads(robot_io.journal.getvalue())
    assert event["proposed"] != event["applied"]


def test_public_snapshot_encodes_cameras_and_omits_model_inputs():
    result = public_snapshot(
        {"id": 1, "raw": {"camera": np.zeros((4, 4, 3), dtype=np.uint8), "_processed": {}}}
    )
    assert "raw" not in result
    assert result["images"]["camera"].startswith("data:image/jpeg;base64,")
    assert "_processed" not in result["images"]
