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
from unittest.mock import patch

import pytest

from lerobot.common.datasets.populate_dataset import add_frame, init_dataset
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.control_robot import calibrate, record, replay, teleoperate
from lerobot.scripts.train import make_optimizer_and_scheduler
from tests.test_robots import make_robot
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, TEST_ROBOT_TYPES, mock_calibration_dir, require_robot


@pytest.mark.parametrize("robot_type, mock", TEST_ROBOT_TYPES)
@require_robot
def test_teleoperate(tmpdir, request, robot_type, mock):
    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        tmpdir = Path(tmpdir)
        calibration_dir = tmpdir / robot_type
        mock_calibration_dir(calibration_dir)
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
    calibrate(robot, arms=robot.available_arms)
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
        mock_calibration_dir(calibration_dir)
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
    tmpdir = Path(tmpdir)

    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = tmpdir / robot_type
        mock_calibration_dir(calibration_dir)
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False or for aloha
        overrides = None

    env_name = "koch_real"
    policy_name = "act_koch_real"

    root = tmpdir / "data"
    repo_id = "lerobot/debug"
    eval_repo_id = "lerobot/eval_debug"

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
    dataset = record(
        robot,
        root,
        repo_id,
        fps=1,
        warmup_time_s=1,
        episode_time_s=1,
        reset_time_s=1,
        num_episodes=2,
        push_to_hub=False,
        # TODO(rcadene, aliberts): test video=True
        video=False,
        # TODO(rcadene): display cameras through cv2 sometimes crashes on mac
        display_cameras=False,
        play_sounds=False,
    )
    assert dataset.num_episodes == 2
    assert len(dataset) == 2

    replay(robot, episode=0, fps=1, root=root, repo_id=repo_id, play_sounds=False)

    # TODO(rcadene, aliberts): rethink this design
    if robot_type == "aloha":
        env_name = "aloha_real"
        policy_name = "act_aloha_real"
    elif robot_type in ["koch", "koch_bimanual"]:
        env_name = "koch_real"
        policy_name = "act_koch_real"
    elif robot_type == "so100":
        env_name = "so100_real"
        policy_name = "act_so100_real"
    elif robot_type == "moss":
        env_name = "moss_real"
        policy_name = "act_moss_real"
    else:
        raise NotImplementedError(robot_type)

    overrides = [
        f"env={env_name}",
        f"policy={policy_name}",
        f"device={DEVICE}",
    ]

    if robot_type == "koch_bimanual":
        overrides += ["env.state_dim=12", "env.action_dim=12"]

    overrides += ["wandb.enable=false"]
    overrides += ["env.fps=1"]

    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=overrides,
    )

    policy = make_policy(hydra_cfg=cfg, dataset_stats=dataset.stats)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    out_dir = tmpdir / "logger"
    logger = Logger(cfg, out_dir, wandb_job_name="debug")
    logger.save_checkpoint(
        0,
        policy,
        optimizer,
        lr_scheduler,
        identifier=0,
    )
    pretrained_policy_name_or_path = out_dir / "checkpoints/last/pretrained_model"

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
        root,
        eval_repo_id,
        pretrained_policy_name_or_path,
        warmup_time_s=1,
        episode_time_s=1,
        reset_time_s=1,
        num_episodes=2,
        run_compute_stats=False,
        push_to_hub=False,
        video=False,
        display_cameras=False,
        play_sounds=False,
        num_image_writer_processes=num_image_writer_processes,
    )

    assert dataset.num_episodes == 2
    assert len(dataset) == 2

    del robot


@pytest.mark.parametrize("robot_type, mock", [("koch", True)])
@require_robot
def test_resume_record(tmpdir, request, robot_type, mock):
    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = tmpdir / robot_type
        mock_calibration_dir(calibration_dir)
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False or for aloha
        overrides = []

    robot = make_robot(robot_type, overrides=overrides, mock=mock)

    root = Path(tmpdir) / "data"
    repo_id = "lerobot/debug"

    dataset = record(
        robot,
        root,
        repo_id,
        fps=1,
        warmup_time_s=0,
        episode_time_s=1,
        num_episodes=1,
        push_to_hub=False,
        video=False,
        display_cameras=False,
        play_sounds=False,
        run_compute_stats=False,
    )
    assert len(dataset) == 1, "`dataset` should contain only 1 frame"

    init_dataset_return_value = {}

    def wrapped_init_dataset(*args, **kwargs):
        nonlocal init_dataset_return_value
        init_dataset_return_value = init_dataset(*args, **kwargs)
        return init_dataset_return_value

    with patch("lerobot.scripts.control_robot.init_dataset", wraps=wrapped_init_dataset):
        dataset = record(
            robot,
            root,
            repo_id,
            fps=1,
            warmup_time_s=0,
            episode_time_s=1,
            num_episodes=2,
            push_to_hub=False,
            video=False,
            display_cameras=False,
            play_sounds=False,
            run_compute_stats=False,
        )
        assert len(dataset) == 2, "`dataset` should contain only 1 frame"
        assert (
            init_dataset_return_value["num_episodes"] == 2
        ), "`init_dataset` should load the previous episode"


@pytest.mark.parametrize("robot_type, mock", [("koch", True)])
@require_robot
def test_record_with_event_rerecord_episode(tmpdir, request, robot_type, mock):
    if mock and robot_type != "aloha":
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = tmpdir / robot_type
        mock_calibration_dir(calibration_dir)
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False or for aloha
        overrides = []

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
    with (
        patch("lerobot.scripts.control_robot.init_keyboard_listener") as mock_listener,
        patch("lerobot.common.robot_devices.control_utils.add_frame", wraps=add_frame) as mock_add_frame,
    ):
        mock_events = {}
        mock_events["exit_early"] = True
        mock_events["rerecord_episode"] = True
        mock_events["stop_recording"] = False
        mock_listener.return_value = (None, mock_events)

        root = Path(tmpdir) / "data"
        repo_id = "lerobot/debug"

        dataset = record(
            robot,
            root,
            repo_id,
            fps=1,
            warmup_time_s=0,
            episode_time_s=1,
            num_episodes=1,
            push_to_hub=False,
            video=False,
            display_cameras=False,
            play_sounds=False,
            run_compute_stats=False,
        )

        assert not mock_events["rerecord_episode"], "`rerecord_episode` wasn't properly reset to False"
        assert not mock_events["exit_early"], "`exit_early` wasn't properly reset to False"
        assert mock_add_frame.call_count == 2, "`add_frame` should have been called 2 times"
        assert len(dataset) == 1, "`dataset` should contain only 1 frame"


@pytest.mark.parametrize("robot_type, mock", [("koch", True)])
@require_robot
def test_record_with_event_exit_early(tmpdir, request, robot_type, mock):
    if mock:
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = tmpdir / robot_type
        mock_calibration_dir(calibration_dir)
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False or for aloha
        overrides = []

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
    with (
        patch("lerobot.scripts.control_robot.init_keyboard_listener") as mock_listener,
        patch("lerobot.common.robot_devices.control_utils.add_frame", wraps=add_frame) as mock_add_frame,
    ):
        mock_events = {}
        mock_events["exit_early"] = True
        mock_events["rerecord_episode"] = False
        mock_events["stop_recording"] = False
        mock_listener.return_value = (None, mock_events)

        root = Path(tmpdir) / "data"
        repo_id = "lerobot/debug"

        dataset = record(
            robot,
            fps=2,
            root=root,
            repo_id=repo_id,
            warmup_time_s=0,
            episode_time_s=1,
            num_episodes=1,
            push_to_hub=False,
            video=False,
            display_cameras=False,
            play_sounds=False,
            run_compute_stats=False,
        )

        assert not mock_events["exit_early"], "`exit_early` wasn't properly reset to False"
        assert mock_add_frame.call_count == 1, "`add_frame` should have been called 1 time"
        assert len(dataset) == 1, "`dataset` should contain only 1 frame"


@pytest.mark.parametrize(
    "robot_type, mock, num_image_writer_processes", [("koch", True, 0), ("koch", True, 1)]
)
@require_robot
def test_record_with_event_stop_recording(tmpdir, request, robot_type, mock, num_image_writer_processes):
    if mock:
        request.getfixturevalue("patch_builtins_input")

        # Create an empty calibration directory to trigger manual calibration
        # and avoid writing calibration files in user .cache/calibration folder
        calibration_dir = tmpdir / robot_type
        mock_calibration_dir(calibration_dir)
        overrides = [f"calibration_dir={calibration_dir}"]
    else:
        # Use the default .cache/calibration folder when mock=False or for aloha
        overrides = []

    robot = make_robot(robot_type, overrides=overrides, mock=mock)
    with (
        patch("lerobot.scripts.control_robot.init_keyboard_listener") as mock_listener,
        patch("lerobot.common.robot_devices.control_utils.add_frame", wraps=add_frame) as mock_add_frame,
    ):
        mock_events = {}
        mock_events["exit_early"] = True
        mock_events["rerecord_episode"] = False
        mock_events["stop_recording"] = True
        mock_listener.return_value = (None, mock_events)

        root = Path(tmpdir) / "data"
        repo_id = "lerobot/debug"

        dataset = record(
            robot,
            root,
            repo_id,
            fps=1,
            warmup_time_s=0,
            episode_time_s=1,
            num_episodes=2,
            push_to_hub=False,
            video=False,
            display_cameras=False,
            play_sounds=False,
            run_compute_stats=False,
            num_image_writer_processes=num_image_writer_processes,
        )

        assert not mock_events["exit_early"], "`exit_early` wasn't properly reset to False"
        assert mock_add_frame.call_count == 1, "`add_frame` should have been called 1 time"
        assert len(dataset) == 1, "`dataset` should contain only 1 frame"
