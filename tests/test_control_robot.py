from pathlib import Path

import pytest

from lerobot import available_robots
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.control_robot import calibrate, record, replay, teleoperate
from tests.test_robots import make_robot
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, require_robot


@pytest.mark.parametrize("robot_type", available_robots)
@require_robot
def test_teleoperate(request, robot_type):
    robot = make_robot(robot_type)
    teleoperate(robot, teleop_time_s=1)
    teleoperate(robot, fps=30, teleop_time_s=1)
    teleoperate(robot, fps=60, teleop_time_s=1)
    del robot


@pytest.mark.parametrize("robot_type", available_robots)
@require_robot
def test_calibrate(request, robot_type):
    robot = make_robot(robot_type)
    calibrate(robot)
    del robot


@pytest.mark.parametrize("robot_type", available_robots)
@require_robot
def test_record_without_cameras(tmpdir, request, robot_type):
    root = Path(tmpdir)
    repo_id = "lerobot/debug"

    robot = make_robot(robot_type, overrides=["~cameras"])
    record(robot, fps=30, root=root, repo_id=repo_id, warmup_time_s=1, episode_time_s=1, num_episodes=2)


@pytest.mark.parametrize("robot_type", available_robots)
@require_robot
def test_record_and_replay_and_policy(tmpdir, request, robot_type):
    env_name = "koch_real"
    policy_name = "act_koch_real"

    root = Path(tmpdir)
    repo_id = "lerobot/debug"

    robot = make_robot(robot_type)
    dataset = record(
        robot, fps=30, root=root, repo_id=repo_id, warmup_time_s=1, episode_time_s=1, num_episodes=2
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

    record(robot, policy, cfg, run_time_s=1)

    del robot
