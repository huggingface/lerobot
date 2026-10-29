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

"""A simulated environment driven through the Robot interface, locally or served."""

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from lerobot.envs.utils import validate_checkpoint
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.transport.zenoh import ZenohConfig
from tests.fixtures.toy_env import ToyEnv, ToyEnvConfig


def test_sim_robot_env_choices_exclude_envs_built_around_a_robot():
    import sys
    from dataclasses import field

    import draccus

    from lerobot.robots import RobotConfig

    @dataclass
    class Wrapper:
        robot: RobotConfig = field(default_factory=lambda: SimRobotConfig(env=ToyEnvConfig()))

    # gym_manipulator embeds a RobotConfig: offering it here would make draccus expand robot and env
    # choices into each other forever.
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(300)
    try:
        args = ["--robot.type=sim", "--robot.env.type=toy"]
        cfg = draccus.parse(Wrapper, args=args)
    finally:
        sys.setrecursionlimit(limit)
    assert isinstance(cfg.robot.env, ToyEnvConfig)
    with pytest.raises(SystemExit):
        draccus.parse(Wrapper, args=["--robot.type=sim", "--robot.env.type=gym_manipulator"])


def test_sim_robot_registration(tmp_path):
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        assert isinstance(robot, SimRobot)
    finally:
        robot.disconnect()


@pytest.mark.parametrize("where", ["local", "served"])
def test_sim_robot_advances_once_per_action(where, request, tmp_path):
    if where == "served":
        config = SimRobotConfig(env=ToyEnvConfig(endpoint=request.getfixturevalue("sim_server")[1]))
    else:
        config = SimRobotConfig(env=ToyEnvConfig())
    config.calibration_dir = tmp_path
    robot = SimRobot(config)
    try:
        robot.connect()
        action = dict.fromkeys(robot.action_features, 0.1)
        assert robot.send_action(action) == pytest.approx(action)
        assert robot.latest.elapsed_steps[0] == 1
        state = [name for name, kind in robot.observation_features.items() if kind is float]
        before = robot.get_observation()
        # Neither waiting nor holding advances the simulation.
        time.sleep(0.1)
        robot.hold()
        after = robot.get_observation()
        assert [after[name] for name in state] == [before[name] for name in state]
        assert robot.latest.elapsed_steps[0] == 1
        robot.send_action(action)
        assert robot.latest.elapsed_steps[0] == 2
    finally:
        robot.disconnect()


def test_rollout_holds_a_sim_robot_with_its_own_hold(tmp_path):
    pytest.importorskip("datasets", reason="lerobot.rollout imports the dataset integration")
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_hold()
    try:
        robot.connect()
        wrapper.send_action({"command_0": 0.5, "command_1": 0.0})
        wrapper.hold()
        assert wrapper.supports_hold and robot.latest.elapsed_steps[0] == 1
    finally:
        robot.disconnect()


@pytest.mark.parametrize("where", ["local", "served"])
def test_a_sim_episode_ends_and_restarts_as_in_gymnasium(where, request, tmp_path):
    config = ToyEnvConfig(endpoint=request.getfixturevalue("sim_server")[1] if where == "served" else None)
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    try:
        robot.connect()
        assert not robot.episode_ended
        # The toy terminates once x passes 1: one step of 2 ends the episode, which keeps its last state.
        robot.send_action({"command_0": 2.0, "command_1": 0.0})
        assert robot.episode_ended
        assert robot.latest.terminated[0] and robot.latest.is_success[0] and robot.latest.reward[0] == 1
        assert robot.get_observation()["state_0"] >= 1
        # As in Gymnasium's autoreset, the next action starts the next episode instead of moving the robot.
        robot.send_action({"command_0": 2.0, "command_1": 0.0})
        assert not robot.episode_ended and robot.latest.elapsed_steps[0] == 0
        assert robot.get_observation()["state_0"] < 1
        # It truncates after 5 steps.
        for _ in range(5):
            robot.send_action({"command_0": 0.0, "command_1": 0.0})
        assert robot.episode_ended and robot.latest.truncated[0] and robot.latest.elapsed_steps[0] == 5
        # A caller running episodes starts each one, from a seed when given.
        robot.start_episode(seed=3)
        start = robot.get_observation()["state_0"]
        robot.start_episode(seed=3)
        assert not robot.episode_ended and robot.get_observation()["state_0"] == start
    finally:
        robot.disconnect()


def test_episode_functions_leave_real_robots_alone():
    from unittest.mock import MagicMock

    from lerobot.robots import Robot
    from lerobot.robots.sim_robot import episode_ended, start_episode

    robot = MagicMock(spec=Robot)
    start_episode(robot)
    assert not episode_ended(robot)
    assert not robot.method_calls


def test_sim_robot_names_the_actions_it_takes(tmp_path):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        with pytest.raises(ValueError, match=r"missing \['command_1'\], unknown \['shoulder.pos'\]"):
            robot.send_action({"command_0": 0.0, "shoulder.pos": 0.0})
    finally:
        robot.disconnect()


def test_sim_robot_actions_follow_controller_order(sim_server, tmp_path):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=sim_server[1]), calibration_dir=tmp_path))
    try:
        assert (
            tuple(robot.action_features)
            == robot.descriptor.action_feature.names
            == ("command_0", "command_1")
        )
        robot.connect()
        assert robot.send_action({"command_1": 0.75, "command_0": 0.25}) == {
            "command_0": 0.25,
            "command_1": 0.75,
        }
    finally:
        robot.disconnect()


def test_sim_robot_records_the_body_its_env_declares(sim_server, tmp_path):
    from lerobot.env_server.protocol import decode_descriptor, encode_descriptor

    for declared, expected in ((None, "sim"), ("so_follower", "so_follower")):
        robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(robot_type=declared), calibration_dir=tmp_path))
        try:
            assert (robot.name, robot.robot_type) == ("sim", expected)
            # A newer server may declare more: this client keeps what it knows.
            wire = {**encode_descriptor(robot.descriptor), "joint_units": ["rad", "rad"]}
            assert decode_descriptor(wire).robot_type == robot.descriptor.robot_type
        finally:
            robot.disconnect()
    # Served, a server simulating another body is refused: its datasets would record the wrong robot.
    config = ToyEnvConfig(robot_type="so_follower", endpoint=sim_server[1])
    with pytest.raises(ValueError, match="robot_type: server ''"):
        SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))


def test_sim_robot_records_scalar_state_actions_and_rgb(sim_server, tmp_path):
    pytest.importorskip("datasets", reason="lerobot.rollout imports the dataset integration")
    from types import SimpleNamespace

    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.rollout import context
    from lerobot.rollout.configs import EpisodicStrategyConfig
    from lerobot.utils.feature_utils import build_dataset_frame

    robot = SimRobot(
        SimRobotConfig(env=ToyEnvConfig(endpoint=sim_server[1]), calibration_dir=tmp_path / "calibration")
    )
    dataset = None
    try:
        robot.connect()
        processors = context._resolve_robot_processors(None, None, None)
        features = context._aggregate_rollout_features(
            processors,
            robot.action_features,
            robot.observation_features,
            use_videos=False,
        )
        cfg = SimpleNamespace(
            resume=False,
            strategy=EpisodicStrategyConfig(),
            dataset=DatasetRecordConfig(
                repo_id="local/rollout_sim_test",
                root=tmp_path / "dataset",
                fps=20,
                video=False,
                push_to_hub=False,
                num_image_writer_threads_per_camera=1,
            ),
        )
        dataset = context._build_rollout_dataset(cfg, robot, features)
        assert dataset.writer.image_writer is not None
        observation = robot.get_observation()
        action = dict.fromkeys(robot.action_features, 0.0)
        frame = {
            **build_dataset_frame(features, observation, "observation"),
            **build_dataset_frame(features, action, "action"),
            "task": "Move to the target",
        }
        dataset.add_frame(frame)
        dataset.save_episode()
        dataset.finalize()
        reader = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
        sample = reader[0]
        np.testing.assert_allclose(
            sample["observation.state"].numpy(), [observation[n] for n in robot._state.names]
        )
        np.testing.assert_array_equal(sample["action"].numpy(), [0, 0])
        image = sample["observation.images.image"].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(image * 255, observation["image"], atol=1e-5)
        assert sample["task"] == "Move to the target"
    finally:
        if dataset is not None:
            dataset.finalize()
        robot.disconnect()


def test_rollout_rejects_checkpoint_schema_before_opening_environment(sim_server, tmp_path, monkeypatch):
    pytest.importorskip("datasets", reason="lerobot.rollout imports the dataset integration")
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.rollout import context

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=sim_server[1]), calibration_dir=tmp_path))
    monkeypatch.setattr(context, "make_robot_from_config", lambda _: robot)
    cfg = SimpleNamespace(
        robot=robot.config,
        teleop=None,
        rename_map={},
        fps=robot.descriptor.fps,
        interpolation_multiplier=1,
        policy=SimpleNamespace(
            input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        ),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        context._connect_rollout_hardware(cfg)
    # A control rate other than the simulator's would replay actions too fast or too slow.
    cfg.fps = robot.descriptor.fps * 2
    with pytest.raises(ValueError, match=f"control rate is {robot.descriptor.fps:g} Hz: pass --fps="):
        context._connect_rollout_hardware(cfg)
    # Each interpolated command advances the simulation too: the loop runs at the rate divided by them.
    cfg.fps, cfg.interpolation_multiplier = robot.descriptor.fps / 2, 2
    with pytest.raises(ValueError, match="shape mismatch"):
        context._connect_rollout_hardware(cfg)
    cfg.fps = robot.descriptor.fps
    with pytest.raises(ValueError, match=f"pass --fps={robot.descriptor.fps / 2:g}"):
        context._connect_rollout_hardware(cfg)
    assert not sim_server[0].sessions
    assert not robot.is_connected


@pytest.mark.parametrize("mode", ["sync", "rtc"])
def test_sim_robot_local_inference_engines(sim_server, tmp_path, mode):
    pytest.importorskip("datasets", reason="lerobot.rollout imports the dataset integration")
    import torch

    from lerobot.inference import RTCInferenceEngine
    from lerobot.inference.sync import SyncInferenceEngine
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.inference.rtc_helpers import ControlledPolicy, Pipeline
    from tests.utils import wait_for

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=sim_server[1]), calibration_dir=tmp_path))
    wrapper = ThreadSafeRobot(robot)
    names = list(robot.action_features)
    state_names = list(robot._state.names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": state_names},
        "action": {"dtype": "float32", "shape": (2,), "names": names},
    }
    policy = ControlledPolicy()
    policy.select_action = lambda batch: torch.tensor([[0.25, 0]], dtype=torch.float32)
    policy.drop_queued_actions = lambda: None
    if mode == "sync":
        engine = SyncInferenceEngine(policy, Pipeline(), Pipeline(), features, names, "reach", "cpu", "sim")
    else:
        engine = RTCInferenceEngine(
            policy,
            Pipeline(),
            Pipeline(),
            robot_wrapper=wrapper,
            rtc_config=RTCConfig(enabled=True, execution_horizon=4),
            dataset_features=features,
            task="reach",
            fps=20,
            device="cpu",
        )
    try:
        robot.connect()
        wrapper.configure_hold()
        engine.start()
        engine.resume()
        observation = wrapper.get_observation()
        if mode == "sync":
            from lerobot.utils.feature_utils import build_dataset_frame

            action = engine.get_action(build_dataset_frame(features, observation, prefix="observation"))
        else:
            engine.notify_observation(observation)
            policy.release.release()
            assert wait_for(lambda: engine.action_queue.qsize() > 0)
            action = engine.get_action(None)
        assert action.shape == (2,)
        wrapper.send_action(dict(zip(names, action.tolist(), strict=True)))
        wrapper.hold()
        engine.pause()
        engine.reset()
        if mode == "rtc":
            assert engine.action_queue.qsize() == 0
    finally:
        policy.release.release(20)
        engine.stop()
        robot.disconnect()


def test_sim_robot_remote_inference_engine(sim_server, tmp_path):
    pytest.importorskip("datasets", reason="lerobot.rollout imports the dataset integration")
    import torch

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.inference import PolicyRunner, RemoteInferenceConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.remote_inference.client import RemoteClient
    from lerobot.remote_inference.engine import RemoteInferenceEngine
    from lerobot.remote_inference.server import PolicyServer, SessionWorker
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.transport.zenoh import ZenohTransport
    from tests.inference.test_policy_runner import ConformingPolicy, tiny_config
    from tests.utils import wait_for

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=sim_server[1]), calibration_dir=tmp_path))
    wrapper = ThreadSafeRobot(robot)
    state = next(f for f in robot.descriptor.features if f.name == "observation.state")
    config = tiny_config()
    config.input_features = {state.name: PolicyFeature(type=FeatureType.STATE, shape=state.shape)}
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    policy = ConformingPolicy(config)
    policy.predict_action_chunk = lambda batch, **kwargs: torch.zeros(1, 8, 2)
    runner = PolicyRunner(
        policy,
        *make_act_pre_post_processors(config),
        action_interval=1 / 20,
        features=(state,),
        action_feature=robot.descriptor.action_feature,
    )
    worker = SessionWorker(
        runner, deployment="sim-policy", artifact_identity="test", semantics=robot.descriptor.semantics
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    policy_server = PolicyServer(worker, transport)
    ready = threading.Event()
    declare_token = transport.declare_token

    def advertise(key):
        token = declare_token(key)
        ready.set()
        return token

    transport.declare_token = advertise
    thread = threading.Thread(target=policy_server.serve)
    thread.start()
    engine = None
    client = None
    try:
        assert ready.wait(5)
        remote_config = RemoteInferenceConfig(
            endpoint=endpoint,
            deployment="sim-policy",
            semantics=robot.descriptor.semantics,
            max_observation_age_s=5,
        )
        client = RemoteClient.connect(remote_config)
        client.admit(
            features=(state,),
            action_feature=robot.descriptor.action_feature,
            semantics=robot.descriptor.semantics,
            action_interval=1 / 20,
            mode="chunk",
        )
        robot.connect()
        wrapper.configure_hold()
        features = {state.name: {"dtype": "float32", "shape": state.shape, "names": state.names}}
        engine = RemoteInferenceEngine(client, remote_config, features, {}, wrapper, "Move to the target")
        engine.start()
        engine.resume()
        engine.notify_observation(wrapper.get_observation())
        assert wait_for(lambda: engine.get_action(None) is not None)
        wrapper.send_action(dict.fromkeys(robot.action_features, 0.0))
        wrapper.hold()
        engine.pause()
        engine.reset()
        assert not engine.failed
    finally:
        if engine is not None:
            engine.stop()
        elif client is not None:
            client.close()
        robot.disconnect()
        policy_server.stop()
        thread.join(5)
        transport.close()
        worker.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("where", ["local", "served"])
def test_recorded_sim_episodes_can_train_and_reload_a_policy(where, request, tmp_path, monkeypatch, clock):
    pytest.importorskip("datasets", reason="lerobot.rollout imports the dataset integration")
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.scripts import lerobot_record
    from lerobot.teleoperators import Teleoperator
    from tests.mocks.mock_teleop import MockTeleopConfig

    env = ToyEnvConfig(endpoint=request.getfixturevalue("sim_server")[1] if where == "served" else None)
    robot = SimRobot(SimRobotConfig(env=env, calibration_dir=tmp_path / "calibration"))
    teleop = MagicMock(spec=Teleoperator)
    teleop.get_action.return_value = {"command_0": 2.0, "command_1": 0.0}
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(lerobot_record, "init_logging", lambda: None)
    monkeypatch.setattr(lerobot_record, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(lerobot_record, "time", clock)
    monkeypatch.setattr(
        lerobot_record,
        "init_keyboard_listener",
        lambda: (None, {"exit_early": False, "stop_recording": False, "rerecord_episode": False}),
    )
    cfg = lerobot_record.RecordConfig(
        robot=robot.config,
        teleop=MockTeleopConfig(),
        play_sounds=False,
        dataset=DatasetRecordConfig(
            repo_id="local/sim_demo",
            root=tmp_path / "dataset",
            single_task="Move to the target",
            fps=20,
            num_episodes=2,
            episode_time_s=0.1,
            reset_time_s=0.1,
            video=False,
            push_to_hub=False,
            no_stamp=True,
        ),
    )
    dataset = lerobot_record.record(cfg)
    # A recorded episode ends with the simulator's: here its first action succeeds, so each holds one frame,
    # recorded from the start of a new world. The unrecorded reset phase still runs between them.
    assert dataset.num_episodes == 2
    reader = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
    assert len(reader) == dataset.num_episodes
    assert teleop.get_action.call_count > len(reader)
    samples = [reader[index] for index in range(len(reader))]
    assert all(sample["observation.state"][0] < 1 for sample in samples)
    checkpoint_config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    validate_checkpoint(robot.descriptor, checkpoint_config, {})
    # Native benchmark camera names remain unchanged.
    assert "observation.image" in env.features
    legacy_config = SimpleNamespace(
        input_features={"observation.image": checkpoint_config.input_features["observation.images.image"]},
        output_features=checkpoint_config.output_features,
    )
    validate_checkpoint(robot.descriptor, legacy_config, {"observation.images.image": "observation.image"})

    pytest.importorskip("torchvision")
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy(
        ACTConfig(
            input_features=checkpoint_config.input_features,
            output_features=checkpoint_config.output_features,
            device="cpu",
            pretrained_backbone_weights=None,
            chunk_size=1,
            n_action_steps=1,
            dim_model=16,
            n_heads=4,
            dim_feedforward=32,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
        )
    )
    batch = {name: torch.stack([sample[name] for sample in samples]) for name in policy.config.input_features}
    batch["action"] = torch.stack([sample["action"] for sample in samples]).unsqueeze(1)
    batch["action_is_pad"] = torch.zeros((len(samples), 1), dtype=torch.bool)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "policy"
    policy.save_pretrained(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint)
    validate_checkpoint(robot.descriptor, policy.config, {})
    try:
        robot.connect()
        for _ in range(5):
            observation = robot.get_observation()
            action = policy.select_action(
                {
                    "observation.state": torch.tensor([[observation["state_0"], observation["state_1"]]]),
                    "observation.images.image": torch.from_numpy(observation["image"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255,
                }
            )
            robot.send_action(dict(zip(robot.action_features, action[0].tolist(), strict=True)))
    finally:
        robot.disconnect()


def test_sim_robot_builds_its_simulator_once(tmp_path):
    before = ToyEnv.builds
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        robot.send_action({"command_0": 0.1, "command_1": 0.0})
        assert ToyEnv.builds - before == 1
    finally:
        robot.disconnect()
