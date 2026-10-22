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

"""Run the existing DAgger intervention path on local and served SimRobot instances."""

from collections.abc import Iterator
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset
from lerobot.inference import InferenceEngine
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import make_default_processors
from lerobot.robots import Robot
from lerobot.robots.sim import SimRobot, SimRobotConfig
from lerobot.rollout.configs import DAggerStrategyConfig, RolloutConfig
from lerobot.rollout.context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RuntimeContext,
)
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.dagger import DAggerPhase, DAggerStrategy
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.feature_utils import hw_to_dataset_features
from tests.env_server.toy_env import ToyEnvConfig
from tests.inference.rtc_helpers import ControlledPolicy, Pipeline, sim_rtc
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig
from tests.utils import wait_for


@pytest.fixture(params=["local", "server"])
def sim_robot(request, tmp_path) -> Iterator[SimRobot]:
    endpoint = request.getfixturevalue("server")[1] if request.param == "server" else None
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=endpoint), seed=42, calibration_dir=tmp_path))
    try:
        robot.connect()
        yield robot
    finally:
        robot.disconnect()


@pytest.mark.parametrize("record_autonomous", [False, True])
def test_dagger_records_takeover_and_resumes_without_extra_sim_steps(
    sim_robot, record_autonomous, tmp_path, monkeypatch
):
    reset_episode = MagicMock(wraps=sim_robot.reset_episode)
    monkeypatch.setattr(sim_robot, "reset_episode", reset_episode)
    wrapper = ThreadSafeRobot(sim_robot)
    wrapper.configure_hold()
    teleop = MockTeleop(MockTeleopConfig())
    correction = dict(zip(sim_robot.action_features, (0.02, 0.0), strict=True))
    monkeypatch.setattr(teleop, "get_action", lambda: correction)
    features = {
        **hw_to_dataset_features(sim_robot.observation_features, "observation", use_video=False),
        **hw_to_dataset_features(sim_robot.action_features, "action"),
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    config = RolloutConfig(
        robot=sim_robot.config,
        teleop=teleop.config,
        strategy=DAggerStrategyConfig(record_autonomous=record_autonomous, num_episodes=2),
        policy=ACTConfig(device="cpu"),
        fps=20,
        duration=2.0,
        play_sounds=False,
        return_to_initial_position=False,
        dataset=DatasetRecordConfig(
            repo_id="local/sim_intervention",
            root=tmp_path / "dataset",
            fps=20,
            num_episodes=2,
            video=False,
            push_to_hub=False,
        ),
    )
    dataset = LeRobotDataset.create(
        config.dataset.repo_id, fps=20, features=features, root=config.dataset.root, use_videos=False
    )
    policy = ControlledPolicy()
    policy.select_action = lambda batch: torch.tensor([[0.01, 0.0]])
    engine = SyncInferenceEngine(
        policy,
        Pipeline(),
        Pipeline(),
        features,
        list(sim_robot.action_features),
        "Move to the target",
        "cpu",
        "sim",
    )
    processors = make_default_processors()
    context = RolloutContext(
        RuntimeContext(config, Event()),
        HardwareContext(wrapper, teleop),
        PolicyContext(None, None, None, engine),
        ProcessorContext(*processors),
        DatasetContext(dataset, features, ordered_action_keys=list(sim_robot.action_features)),
    )
    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=record_autonomous, num_episodes=2))
    strategy._engine = engine
    strategy._interpolator = ActionInterpolator()
    strategy._episode_duration_s = 1e9
    events = strategy._events
    read_observation, send_action = wrapper.get_observation, wrapper.send_action
    pause_count = 0
    commands = []

    def observe():
        nonlocal pause_count
        observation = read_observation()
        if events.phase == DAggerPhase.PAUSED:
            pause_count += 1
            events.request_transition("correction" if pause_count == 1 else "pause_resume")
        return observation

    def send(action):
        result = send_action(action)
        commands.append(result)
        if len(commands) <= 2:
            event = "correction" if events.phase == DAggerPhase.CORRECTING else "pause_resume"
            events.request_transition(event)
        else:
            context.runtime.shutdown_event.set()
        return result

    monkeypatch.setattr(wrapper, "get_observation", observe)
    monkeypatch.setattr(wrapper, "send_action", send)
    engine.start()
    try:
        strategy.run(context)
        reset_episode.assert_not_called()
        assert sim_robot.episode_status.step == 3
        np.testing.assert_allclose(
            [[action[name] for name in sim_robot.action_features] for action in commands],
            [[0.01, 0.0], [0.02, 0.0], [0.01, 0.0]],
        )
        assert dataset.num_episodes == 1
        assert dataset.num_frames == (3 if record_autonomous else 1)
        flags = [bool(dataset[index]["intervention"].item()) for index in range(dataset.num_frames)]
        assert flags == ([False, True, False] if record_autonomous else [True])
    finally:
        engine.stop()
        dataset.finalize()


def test_rtc_takeover_discards_inflight_prediction_and_resumes_from_human_state(sim_robot):
    with sim_rtc(sim_robot) as rig:
        rig.complete(1)
        assert rig.tick() is not None
        rig.wait_for_call(2)
        strategy = DAggerStrategy(DAggerStrategyConfig())
        strategy._engine = rig.engine
        strategy._interpolator = rig.interpolator
        teleop = MockTeleop(MockTeleopConfig())
        context = SimpleNamespace(hardware=HardwareContext(rig.wrapper, teleop))
        action = dict(zip(sim_robot.action_features, rig.policy.chunks[0][0].tolist(), strict=True))

        def transition(previous, next_phase):
            strategy._apply_transition(
                previous, next_phase, rig.engine, rig.interpolator, teleop, context, action
            )

        transition(DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)
        assert not rig.engine.dispatch_allowed()
        assert sim_robot.episode_status.step == 1
        transition(DAggerPhase.PAUSED, DAggerPhase.CORRECTING)
        assert sim_robot.episode_status.step == 1
        rig.wrapper.send_action(dict(zip(sim_robot.action_features, (0.02, 0.0), strict=True)))
        observation = sim_robot.get_observation()
        transition(DAggerPhase.CORRECTING, DAggerPhase.PAUSED)
        transition(DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)
        rig.capture()
        rig.policy.release.release()
        rig.wait_for_call(3)
        assert rig.engine.action_queue.empty()
        assert rig.policy.calls[2][0] is None
        np.testing.assert_allclose(
            rig.policy.action_observations[2][0].numpy(),
            [observation[name] for name in sim_robot._state_feature.names],
        )
        rig.policy.release.release()
        assert wait_for(lambda: not rig.engine.action_queue.empty()), rig.engine.failure_traceback
        torch.testing.assert_close(rig.tick(), rig.policy.chunks[2][0])
        assert sim_robot.episode_status.step == 3


def test_physical_robot_keeps_the_existing_handover_ramp(monkeypatch):
    teleop = MockTeleop(MockTeleopConfig())
    target = {"motor_1.pos": 0.2}
    previous = {"motor_1.pos": 0.1}
    monkeypatch.setattr(teleop, "get_action", lambda: target)
    wrapper = MagicMock(spec=ThreadSafeRobot)
    wrapper.inner = MagicMock(spec=Robot)
    wrapper.get_observation.return_value = previous
    context = SimpleNamespace(
        hardware=HardwareContext(wrapper, teleop), processors=ProcessorContext(*make_default_processors())
    )
    ramps = []
    monkeypatch.setattr(
        "lerobot.rollout.strategies.dagger.follower_smooth_move_to", lambda *args: ramps.append(args)
    )
    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._apply_transition(
        DAggerPhase.PAUSED,
        DAggerPhase.CORRECTING,
        MagicMock(spec=InferenceEngine),
        ActionInterpolator(),
        teleop,
        context,
        previous,
    )
    assert ramps == [(wrapper, previous, target)]


def test_rollout_entrypoint_forwards_teleop_processor(monkeypatch):
    from lerobot.scripts import lerobot_rollout

    processor = make_default_processors()[0]
    shutdown_event = Event()
    config = SimpleNamespace(
        inference=SimpleNamespace(log_level="INFO"),
        display_data=False,
        interactive=False,
        strategy=SimpleNamespace(type="dagger"),
        robot=None,
        fps=20,
        duration=1,
    )
    context = SimpleNamespace(policy=SimpleNamespace(inference=SimpleNamespace(failed=False)))
    build_context = MagicMock(return_value=context)
    strategy = MagicMock()
    monkeypatch.setattr(lerobot_rollout, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        lerobot_rollout,
        "ProcessSignalHandler",
        lambda **kwargs: SimpleNamespace(shutdown_event=shutdown_event),
    )
    monkeypatch.setattr(lerobot_rollout, "build_rollout_context", build_context)
    monkeypatch.setattr(lerobot_rollout, "create_strategy", lambda cfg: strategy)

    lerobot_rollout.rollout.__wrapped__(config, teleop_action_processor=processor)

    build_context.assert_called_once_with(config, shutdown_event, processor)
    strategy.run.assert_called_once_with(context)
    strategy.teardown.assert_called_once_with(context)


def test_rollout_main_forwards_teleop_processor(monkeypatch):
    from lerobot.scripts import lerobot_rollout

    processor = make_default_processors()[0]
    rollout = MagicMock()
    monkeypatch.setattr(lerobot_rollout, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_rollout, "rollout", rollout)

    lerobot_rollout.main(teleop_action_processor=processor)

    rollout.assert_called_once_with(teleop_action_processor=processor)
