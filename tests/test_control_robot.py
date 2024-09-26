"""
Tests for physical robots and their mocked versions.
If the physical robots are not connected to the computer, or not working,
the test will be skipped.

Example of running a specific test:
```bash
pytest -sx tests/test_control_robot.py::test_teleoperate
```

Example of running test on real robots connected to the computer:
```bash
pytest -sx 'tests/test_control_robot.py::test_teleoperate[koch-False]'
pytest -sx 'tests/test_control_robot.py::test_teleoperate[koch_bimanual-False]'
pytest -sx 'tests/test_control_robot.py::test_teleoperate[aloha-False]'
```

Example of running test on a mocked version of robots:
```bash
pytest -sx 'tests/test_control_robot.py::test_teleoperate[koch-True]'
pytest -sx 'tests/test_control_robot.py::test_teleoperate[koch_bimanual-True]'
pytest -sx 'tests/test_control_robot.py::test_teleoperate[aloha-True]'
```
"""

from pathlib import Path

import pytest

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.control_robot import calibrate, get_available_arms, record, replay, teleoperate
from tests.test_robots import make_robot
from tests.utils import (
    DEFAULT_CONFIG_PATH,
    DEVICE,
    TEST_ROBOT_TYPES,
    mock_robot_or_skip_test_when_not_available,
)


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
def test_teleoperate(monkeypatch, robot_type, mock):
    mock_robot_or_skip_test_when_not_available(monkeypatch, robot_type, mock)

    robot = make_robot(robot_type)
    teleoperate(robot, teleop_time_s=1)
    teleoperate(robot, fps=30, teleop_time_s=1)
    teleoperate(robot, fps=60, teleop_time_s=1)
    del robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
def test_calibrate(monkeypatch, robot_type, mock):
    mock_robot_or_skip_test_when_not_available(monkeypatch, robot_type, mock)

    robot = make_robot(robot_type)
    calibrate(robot, arms=get_available_arms(robot))
    del robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
def test_record_without_cameras(tmpdir, monkeypatch, robot_type, mock):
    mock_robot_or_skip_test_when_not_available(monkeypatch, robot_type, mock)

    root = Path(tmpdir)
    repo_id = "lerobot/debug"

    robot = make_robot(robot_type, overrides=["~cameras"])
    record(
        robot,
        fps=30,
        root=root,
        repo_id=repo_id,
        warmup_time_s=1,
        episode_time_s=1,
        num_episodes=2,
        run_compute_stats=False,
        push_to_hub=False,
        video=False,
    )


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
def test_record_and_replay_and_policy(tmpdir, monkeypatch, robot_type, mock):
    mock_robot_or_skip_test_when_not_available(monkeypatch, robot_type, mock)

    env_name = "koch_real"
    policy_name = "act_koch_real"

    root = Path(tmpdir)
    repo_id = "lerobot/debug"

    robot = make_robot(robot_type)
    dataset = record(
        robot,
        fps=30,
        root=root,
        repo_id=repo_id,
        warmup_time_s=1,
        episode_time_s=1,
        num_episodes=2,
        push_to_hub=False,
        # TODO(rcadene, aliberts): test video=True
        video=False,
    )

    replay(robot, episode=0, fps=30, root=root, repo_id=repo_id)

    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ],
    )

    policy = make_policy(hydra_cfg=cfg, dataset_stats=dataset.stats)

    record(
        robot,
        policy,
        cfg,
        warmup_time_s=1,
        episode_time_s=1,
        num_episodes=2,
        run_compute_stats=False,
        push_to_hub=False,
        video=False,
    )

    del robot
