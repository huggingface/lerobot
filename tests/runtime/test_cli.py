from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from lerobot.runtime.cli import _build_rollout_runtime_io, _parse_args


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
