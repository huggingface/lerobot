

from pathlib import Path
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.control_robot import record_dataset, replay_episode, run_policy, teleoperate
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE


def test_teleoperate():
    robot = make_robot("koch")
    teleoperate(robot, teleop_time_s=1)
    teleoperate(robot, fps=30, teleop_time_s=1)
    teleoperate(robot, fps=60, teleop_time_s=1)
    del robot


def test_record_dataset_and_replay_episode_and_run_policy(tmpdir):
    robot_name = "koch"
    env_name = "koch_real"
    policy_name = "act_real"

    #root = Path(tmpdir)
    root = Path("tmp/data")
    repo_id = "lerobot/debug"

    robot = make_robot(robot_name)
    dataset = record_dataset(robot, fps=30, root=root, repo_id=repo_id, warmup_time_s=2, episode_time_s=2, num_episodes=2)

    replay_episode(robot, episode=0, fps=30, root=root, repo_id=repo_id)

    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
    )

    policy = make_policy(hydra_cfg=cfg, dataset_stats=dataset.stats)

    run_policy(robot, policy, cfg, run_time_s=1)

    del robot

