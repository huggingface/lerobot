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

import multiprocessing
from pathlib import Path

import pytest

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.control_robot import calibrate, get_available_arms, record, replay, teleoperate
from tests.test_robots import make_robot
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, TEST_ROBOT_TYPES, require_robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
@require_robot
def test_teleoperate(tmpdir, request, robot_type, mock):
    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        tmpdir = Path(tmpdir)
        calibration_dir = tmpdir / robot_type
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False
        overrides = None

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
    teleoperate(robot, teleop_time_s=1)
    teleoperate(robot, fps=30, teleop_time_s=1)
    teleoperate(robot, fps=60, teleop_time_s=1)
    del robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
@require_robot
def test_calibrate(tmpdir, request, robot_type, mock):
    if mock:
        request.getfixturevalue("patch_builtins_input")

    # Create an empty calibration directory to trigger manual calibration
    tmpdir = Path(tmpdir)
    calibration_dir = tmpdir / robot_type
    overrides_calibration_dir = [f"calibration_dir={calibration_dir}"]

    robot = make_robot(robot_type, overrides=overrides_calibration_dir, mock=mock)
    calibrate(robot, arms=get_available_arms(robot))
    del robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
@require_robot
def test_record_without_cameras(tmpdir, request, robot_type, mock):
    # Avoid using cameras
    overrides = ["~cameras"]

    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = Path(tmpdir) / robot_type
        overrides.append(f"calibration_dir={calibration_dir}")

    root = Path(tmpdir) / "data"
    repo_id = "lerobot/debug"

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
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
        play_sounds=False,
    )


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
@require_robot
def test_record_and_replay_and_policy(tmpdir, request, robot_type, mock):
    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = Path(tmpdir) / robot_type
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False or for aloha
        overrides = None

    env_name = "koch_real"
    policy_name = "act_koch_real"

    root = Path(tmpdir) / "data"
    repo_id = "lerobot/debug"

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
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
        # TODO(rcadene): display cameras through cv2 sometimes crashes on mac
        display_cameras=False,
        play_sounds=False,
    )

    replay(robot, episode=0, fps=30, root=root, repo_id=repo_id, play_sounds=False)

    # TODO(rcadene, aliberts): rethink this design
    if robot_type == "aloha":
        env_name = "aloha_real"
        policy_name = "act_aloha_real"
    elif robot_type in ["koch", "koch_bimanual"]:
        env_name = "koch_real"
        policy_name = "act_koch_real"
    else:
        raise NotImplementedError(robot_type)

    overrides = [
        f"env={env_name}",
        f"policy={policy_name}",
        f"device={DEVICE}",
    ]

    if robot_type == "koch_bimanual":
        overrides += ["env.state_dim=12", "env.action_dim=12"]

    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=overrides,
    )

    policy = make_policy(hydra_cfg=cfg, dataset_stats=dataset.stats)

    # In `examples/9_use_aloha.md`, we advise using `num_image_writer_processes=1`
    # during inference, to reach constent fps, so we test this here.
    if robot_type == "aloha":
        num_image_writer_processes = 1

        # `multiprocessing.set_start_method("spawn", force=True)` avoids a hanging issue
        # before exiting pytest. However, it outputs the following error in the log:
        # Traceback (most recent call last):
        #     File "<string>", line 1, in <module>
        #     File "/Users/rcadene/miniconda3/envs/lerobot/lib/python3.10/multiprocessing/spawn.py", line 116, in spawn_main
        #         exitcode = _main(fd, parent_sentinel)
        #     File "/Users/rcadene/miniconda3/envs/lerobot/lib/python3.10/multiprocessing/spawn.py", line 126, in _main
        #         self = reduction.pickle.load(from_parent)
        #     File "/Users/rcadene/miniconda3/envs/lerobot/lib/python3.10/multiprocessing/synchronize.py", line 110, in __setstate__
        #         self._semlock = _multiprocessing.SemLock._rebuild(*state)
        # FileNotFoundError: [Errno 2] No such file or directory
        # TODO(rcadene, aliberts): fix FileNotFoundError in multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    else:
        num_image_writer_processes = 0

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
        display_cameras=False,
        play_sounds=False,
        num_image_writer_processes=num_image_writer_processes,
    )

    del robot
