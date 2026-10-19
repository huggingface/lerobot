#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import re
from unittest.mock import MagicMock, PropertyMock, patch

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import RecordConfig, record, record_loop
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.keyboard import KeyboardBaseTeleop, KeyboardTeleop
from lerobot.teleoperators.so_leader import SO100Leader
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def _ticks(summary: str) -> int:
    """Sample size out of a cadence report — every other number is an average over it."""
    return int(re.search(r"(\d+) ticks", summary).group(1))


def _step_calls(summary: str, step: str) -> int:
    """How many ticks ran *step*, off the loop-body breakdown of a run summary."""
    return int(re.search(rf"\n\s+{step}\s+.*· (\d+) calls", summary).group(1))


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate(cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        fps=30,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)

    # A teleop session has no episodes, so there is one cadence block for the whole run,
    # and the steps it names are the ones the loop wraps.
    (summary,) = cadence_log
    assert summary.startswith("Cadence summary — whole run · target 30 Hz (33.3 ms budget per tick):")
    for step in ("observe", "teleop", "send"):
        assert step in summary, step


def test_teleoperate_with_a_base_teleop_alone():
    cfg = TeleoperateConfig(
        robot=MockRobotConfig(),
        base_teleop=MockTeleopConfig(n_motors=1),
        fps=30,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_record_with_a_base_teleop_alone_holds_the_other_actions(tmp_path):
    # The base teleop commands motor_1; with no arm teleop, motor_2 and motor_3 hold their first reading.
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_base",
        num_episodes=1,
        episode_time_s=0.2,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=dataset_cfg,
        base_teleop=MockTeleopConfig(n_motors=1),
        play_sounds=False,
    )

    dataset = record(cfg)

    actions = dataset.hf_dataset["action"]
    assert len(actions) > 1
    assert all(list(a[1:]) == list(actions[0][1:]) for a in actions)
    assert len({float(a[0]) for a in actions}) > 1


def test_record_requires_a_teleop():
    dataset_cfg = DatasetRecordConfig(repo_id=DUMMY_REPO_ID, single_task="Dummy task", push_to_hub=False)
    with pytest.raises(ValueError, match="base_teleop"):
        RecordConfig(robot=MockRobotConfig(), dataset=dataset_cfg)


@pytest.mark.parametrize("with_arm", [False, True])
@pytest.mark.parametrize("recording", [False, True])
def test_mobile_control_delivers_fresh_observations_to_each_device(
    tmp_path, monkeypatch, with_arm, recording
):
    pending = {}
    action_counts = {}

    def on_observation(self, observation):
        assert pending.get(self) is None
        pending[self] = observation

    def get_action(self):
        observation = pending.pop(self)
        assert observation
        action_counts[self] = action_counts.get(self, 0) + 1
        return {
            key: observation[key]
            for key in self.action_features
            if self.config.n_motors == 1 or key != "motor_1.pos"
        }

    monkeypatch.setattr(MockTeleop, "on_observation", on_observation)
    monkeypatch.setattr(MockTeleop, "get_action", get_action)
    kwargs = {
        "robot": MockRobotConfig(),
        "base_teleop": MockTeleopConfig(n_motors=1),
        "teleop": MockTeleopConfig() if with_arm else None,
    }
    if recording:
        dataset_cfg = DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "mobile_observed",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        )
        record(RecordConfig(**kwargs, dataset=dataset_cfg, play_sounds=False))
    else:
        teleoperate(TeleoperateConfig(**kwargs, fps=30, teleop_time_s=0.1))
    assert not pending
    assert len(action_counts) == (2 if with_arm else 1)
    assert all(count > 0 for count in action_counts.values())


def test_record_disconnects_arm_after_base_disconnects(tmp_path):
    arm_cfg, base_cfg = MockTeleopConfig(), MockTeleopConfig(n_motors=1)
    arm, base = MockTeleop(arm_cfg), MockTeleop(base_cfg)
    arm.motors = ["motor_2", "motor_3"]
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "disconnected_base",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(), dataset=dataset_cfg, teleop=arm_cfg, base_teleop=base_cfg, play_sounds=False
    )
    get_action = base.get_action

    def disconnect_after_action():
        action = get_action()
        base.disconnect()
        return action

    with (
        patch("lerobot.scripts.lerobot_record.make_teleoperator_from_config", side_effect=[arm, base]),
        patch.object(base, "get_action", side_effect=disconnect_after_action),
    ):
        record(cfg)
    assert not base.is_connected
    assert not arm.is_connected


def _spy_on_observation():
    """Patches MockTeleop so each tick records whether `on_observation` ran before `get_action`."""
    from tests.mocks.mock_teleop import MockTeleop

    calls: list[str] = []
    get_action = MockTeleop.get_action

    def on_observation(self, observation):
        assert observation and all(key.endswith(".pos") for key in observation)
        calls.append("on_observation")

    def spy_get_action(self):
        calls.append("get_action")
        return get_action(self)

    return calls, patch.multiple(MockTeleop, on_observation=on_observation, get_action=spy_get_action)


def _observed_before_every_action(calls: list[str]) -> bool:
    pairs = list(zip(calls[::2], calls[1::2], strict=False))
    return bool(pairs) and all(pair == ("on_observation", "get_action") for pair in pairs)


def test_teleoperate_hands_the_observation_to_the_teleop():
    calls, spy = _spy_on_observation()
    cfg = TeleoperateConfig(robot=MockRobotConfig(), teleop=MockTeleopConfig(), fps=30, teleop_time_s=0.1)
    with spy:
        teleoperate(cfg)
    assert _observed_before_every_action(calls)


def test_record_hands_the_observation_to_the_teleop(tmp_path):
    calls, spy = _spy_on_observation()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_observed",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(), dataset=dataset_cfg, teleop=MockTeleopConfig(), play_sounds=False
    )
    with spy:
        record(cfg)
    assert _observed_before_every_action(calls)


def test_record_multi_teleop_receives_fresh_observations_before_actions() -> None:
    # Exercise the legacy list entry point through the new wrapper without connecting hardware.
    robot = MagicMock(spec=LeKiwiClient)
    robot.name = "lekiwi_client"
    robot.config = LeKiwiClientConfig(remote_ip="unused", cameras={})
    arm = MagicMock(spec=SO100Leader)
    keyboard = MagicMock(spec=KeyboardBaseTeleop)
    keyboard.name = "keyboard_base"
    observations = [{"arm_shoulder.pos": float(tick)} for tick in range(3)]
    robot.get_observation.side_effect = observations
    events = {"exit_early": False}
    pending: dict[str, RobotObservation] = {}
    received: dict[str, list[RobotObservation]] = {"arm": [], "keyboard": []}
    sent: list[RobotAction] = []

    def observe(device: str, observation: RobotObservation) -> None:
        assert device not in pending
        pending[device] = observation
        received[device].append(observation)

    def arm_action() -> RobotAction:
        observation = pending.pop("arm")
        return {"shoulder.pos": observation["arm_shoulder.pos"] + 10.0}

    def keyboard_action() -> RobotAction:
        observation = pending.pop("keyboard")
        assert observation is received["arm"][-1]
        return {"x.vel": 0.2}

    def send_action(action: RobotAction) -> RobotAction:
        sent.append(action)
        if len(sent) == len(observations):
            events["exit_early"] = True
        return action

    arm.on_observation.side_effect = lambda obs: observe("arm", obs)
    keyboard.on_observation.side_effect = lambda obs: observe("keyboard", obs)
    arm.get_action.side_effect = arm_action
    keyboard.get_action.side_effect = keyboard_action
    robot.send_action.side_effect = send_action
    teleop_processor, action_processor, observation_processor = make_default_processors()

    record_loop(
        robot=robot,
        events=events,
        fps=200,
        teleop_action_processor=teleop_processor,
        robot_action_processor=action_processor,
        robot_observation_processor=observation_processor,
        teleop=[arm, keyboard],
        control_time_s=10.0,
    )

    assert not pending
    for device in received:
        assert len(received[device]) == len(observations)
        assert all(
            actual is expected for actual, expected in zip(received[device], observations, strict=True)
        )
    assert sent == [{"arm_shoulder.pos": float(tick) + 10.0, "x.vel": 0.2} for tick in range(3)]
    assert arm.get_action.call_count == keyboard.get_action.call_count == 3


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)

    # Replay has to hit the dataset's frame rate or the trajectory plays back at the
    # wrong speed, so it reports its cadence like every other loop.  Its block is the
    # last one and names its own steps.
    assert cadence_log[-1].startswith("Cadence summary — whole run · target 30 Hz")
    assert "read_frame" in cadence_log[-1]


def test_record_reports_a_cadence_summary_per_episode_and_for_the_run(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "cadence",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    record(cfg)

    assert len(cadence_log) == 3
    per_episode, run = cadence_log[:2], cadence_log[2]
    assert [m.split(":")[0] for m in per_episode] == ["Cadence (episode 0)", "Cadence (episode 1)"]
    assert run.startswith("Cadence summary — whole run, 2 episodes")
    # Windows partition the session, so the episodes account for every tick of the run...
    assert _ticks(run) == sum(_ticks(m) for m in per_episode)
    # ...and every one of those ticks wrote a frame.  The reset phase paces at the same
    # fps but records nothing, so it runs on its own timer rather than diluting the
    # numbers that answer "did I record at `fps`?".
    assert _step_calls(run, "record") == _step_calls(run, "observe") == _ticks(run)


def test_record_forwards_compressed_images_setting_to_reset_phase(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "compressed_images",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        display_compressed_images=True,
        play_sounds=False,
    )

    with patch("lerobot.scripts.lerobot_record.record_loop", wraps=record_loop) as mock_record_loop:
        record(cfg)

    # Recording episode 0, resetting, then recording episode 1 should all use the same
    # image representation so visualization backends do not receive mixed message types.
    assert [
        call.kwargs.get("display_compressed_images", False) for call in mock_record_loop.call_args_list
    ] == [True, True, True]


def test_record_loop_without_a_teleoperator_paces_and_terminates():
    # Regression: the no-teleop branch used to `continue` past both the pacing sleep and
    # the `timestamp` update, so a reset phase with no teleop device spun as fast as the
    # CPU allowed and never reached `control_time_s` at all.
    robot = make_robot_from_config(MockRobotConfig())
    robot.connect()
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    calls = 0
    real_get_observation = robot.get_observation

    def counted_get_observation():
        nonlocal calls
        calls += 1
        assert calls <= 20, "loop is spinning: 20 iterations of a 0.1 s phase at 30 Hz"
        return real_get_observation()

    robot.get_observation = counted_get_observation

    try:
        record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False, "rerecord_episode": False},
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=None,
            control_time_s=0.1,
        )
    finally:
        robot.disconnect()

    # 0.1 s at 30 Hz is 3 ticks; the upper bound is what proves the phase was paced.
    assert 1 <= calls <= 6


@pytest.mark.parametrize("keyboard_type", ["keyboard", "keyboard_base"])
@pytest.mark.parametrize("hold_uncommanded", [False, True])
def test_lekiwi_keyboard_cli_uses_teleop_mapping(keyboard_type, hold_uncommanded) -> None:
    cfg = draccus.parse(
        TeleoperateConfig,
        args=[
            "--robot.type=lekiwi_client",
            "--robot.remote_ip=unused",
            f"--base_teleop.type={keyboard_type}",
            f"--hold_uncommanded={str(hold_uncommanded).lower()}",
            "--fps=200",
            "--teleop_time_s=0.02",
        ],
    )
    robot = MagicMock(spec=LeKiwiClient)
    robot.config = cfg.robot
    robot.name = "lekiwi_client"
    obs = {"arm_shoulder.pos": 10.0, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    robot.action_features = dict.fromkeys(obs, float)
    robot.get_observation.return_value = obs
    robot._from_keyboard_to_base_action.side_effect = AssertionError("Robot must not map keyboard input")

    def connect_keyboard(self, calibrate=True):
        self.event_queue.put(("w", True))

    with (
        patch("lerobot.scripts.lerobot_teleoperate.make_robot_from_config", return_value=robot),
        patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"),
        patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True),
        patch.object(KeyboardTeleop, "connect", connect_keyboard),
        patch.object(KeyboardBaseTeleop, "connect", connect_keyboard),
    ):
        teleoperate(cfg)

    expected = {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0}
    if hold_uncommanded:
        expected["arm_shoulder.pos"] = 10.0
    assert robot.send_action.call_count > 0
    assert all(call.args[0] == expected for call in robot.send_action.call_args_list)
    robot._from_keyboard_to_base_action.assert_not_called()
