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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.runtime.cli import _ask_runtime, _build_rollout_runtime_io, _parse_args
from lerobot.runtime.language_runtime import RuntimeState


def test_parse_args_preserves_rollout_robot_overrides():
    args = _parse_args(
        [
            "--policy.path=checkpoint",
            "--robot.type=so101_follower",
            "--robot.calibration_dir=/tmp/calibration",
        ]
    )

    assert args.robot_type == "so101_follower"
    assert "--robot.calibration_dir=/tmp/calibration" in args.raw_argv


def test_parse_args_rejects_removed_dataset_replay_flags():
    with pytest.raises(SystemExit):
        _parse_args(["--policy.path=checkpoint", "--dataset.repo_id=dataset"])


def test_rollout_runtime_io_uses_context_processors():
    robot = MagicMock()
    robot.robot_type = "mock_robot"
    robot.cameras = {}
    robot.get_observation.return_value = {"joint.pos": 1.5}
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(robot_wrapper=robot),
        runtime=SimpleNamespace(cfg=SimpleNamespace(device="cpu")),
        processors=SimpleNamespace(
            robot_observation_processor=lambda observation: observation,
            robot_action_processor=lambda pair: pair[0],
        ),
        policy=SimpleNamespace(
            preprocessor=lambda observation: observation,
            postprocessor=lambda action: action,
        ),
        data=SimpleNamespace(
            dataset_features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["joint.pos"],
                },
                "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            }
        ),
    )
    provider, executor = _build_rollout_runtime_io(ctx, rerun_log=False, get_task=lambda: "move")

    observation = provider()
    executor(torch.tensor([[2.0]]))

    assert observation["observation.state"].shape == (1, 1)
    robot.send_action.assert_called_once_with({"joint.pos": 2.0})


def test_ask_runtime_pauses_and_routes_current_observation(capsys):
    adapter = MagicMock()
    adapter.generate_text.return_value = "The mug is beside the bowl."
    runtime = SimpleNamespace(
        state=RuntimeState(mode="action"),
        policy_adapter=adapter,
        _current_observation=lambda: {"image": "current"},
    )
    runtime.state.action_queue.extend([1, 2])

    answer = _ask_runtime(runtime, "What is beside the bowl?")

    assert answer == "The mug is beside the bowl."
    assert runtime.state.mode == "paused"
    assert not runtime.state.action_queue
    adapter.generate_text.assert_called_once_with(
        "vqa",
        {"image": "current"},
        runtime.state,
        user_text="What is beside the bowl?",
    )
    assert "[policy] The mug is beside the bowl." in capsys.readouterr().out
