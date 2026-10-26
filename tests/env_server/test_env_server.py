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

import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from lerobot.env_server.client import EnvClient
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError
from lerobot.transport.zenoh import ZenohConfig
from tests.env_server.toy_env import ToyEnvConfig


def test_torch_free_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
import numpy as np
import lerobot.env_server.server, lerobot.sims.vector_env, lerobot.sims.integrations.envhub
from lerobot.transport.wire.client import query_reply, decode_reply
from lerobot.utils.feature_spec import FeatureSpec, feature_mismatch, validate_array
from lerobot.transport.wire.protocol import Envelope, MessageType, correlate_reply, raise_peer_error, validate_reply
feature = FeatureSpec('state', (2,), 'float32', semantics='state-v1')
assert feature_mismatch(feature, feature) is None
validate_array(np.zeros(2, np.float32), feature)
request = Envelope(MessageType.DESCRIBE, request_id='request')
reply = request.reply(MessageType.ACCEPTED, {})
validate_reply(reply, request, MessageType.ACCEPTED)
for prefix in ('torch', 'lerobot.policies', 'lerobot.remote_inference'):
    assert not any(
        module is not None and (name == prefix or name.startswith(prefix + '.'))
        for name, module in sys.modules.items()
    )
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("use_async_envs", [False, True])
def test_batch_terminal_freeze_and_seed_parity(server, use_async_envs):
    _, endpoint = server
    with_client = EnvClient(endpoint)
    try:
        descriptor = with_client.describe()
        assert descriptor.action_feature.shape == (2,)
        with_client.open(2, use_async_envs=use_async_envs)
        first = with_client.request("reset", seeds=[10, 20])["result"]
        second = with_client.request("reset", seeds=[10, 20])["result"]
        np.testing.assert_array_equal(first["obs"]["observation.state"], second["obs"]["observation.state"])
        ended = with_client.request("step", actions=np.array([[2, 0], [0, 0]], np.float32))["result"]
        assert ended["is_success"].tolist() == [True, False]
        assert ended["reward"].tolist() == [1.0, 0.0]
        frozen = ended["obs"]["observation.state"][0].copy()
        for _ in range(5):
            result = with_client.request("step", actions=np.ones((2, 2), np.float32) * 0.1)["result"]
        np.testing.assert_array_equal(result["obs"]["observation.state"][0], frozen)
        assert (
            result["reward"][0] == 0.0
        )  # The terminal reward is earned once, as with the old freeze wrapper.
        assert result["step"].tolist() == [1, 5]
        assert result["truncated"].tolist() == [False, True]
        frames = with_client.request("render")["frames"]
        assert len(frames) == 2 and frames[0].shape == (8, 8, 3)
    finally:
        with_client.close()


def test_admission_and_stale_generation(server):
    _, endpoint = server
    client = EnvClient(endpoint)
    other = EnvClient(endpoint)
    try:
        with pytest.raises(ProtocolError, match="between 1 and 32"):
            client.open(33)
        client.open(1)
        with pytest.raises(ProtocolError) as error:
            other.open(1)
        assert error.value.code == ErrorCode.BUSY
        client.request("reset")
        client.generation = 0
        with pytest.raises(ProtocolError) as error:
            client.request("step", actions=np.zeros((1, 2), np.float32))
        assert error.value.code == ErrorCode.STALE
        assert client.failed
    finally:
        client.close()
        other.close()


def test_jpeg_observations_match_raw(server):
    from lerobot.env_server.client import BatchedEnv

    _, endpoint = server
    obs = {}
    for encoding in ("raw", "jpeg"):
        env = BatchedEnv(EnvClient(endpoint, encoding=encoding, jpeg_quality=95), num_envs=2)
        try:
            obs[encoding], _ = env.reset(seed=[10, 20])
        finally:
            env.close()
    raw, jpeg = obs["raw"]["observation.image"], obs["jpeg"]["observation.image"]
    assert jpeg.shape == raw.shape == (2, 8, 8, 3) and jpeg.dtype == np.uint8
    assert np.abs(jpeg.astype(int) - raw.astype(int)).mean() < 8
    np.testing.assert_array_equal(obs["jpeg"]["observation.state"], obs["raw"]["observation.state"])
    with pytest.raises(ValueError, match="raw or jpeg"):
        EnvClient(endpoint, encoding="png")


def test_vector_client(server):
    from lerobot.env_server.client import BatchedEnv
    from lerobot.envs.utils import prepare_canonical_observation

    _, endpoint = server
    env = BatchedEnv(EnvClient(endpoint), num_envs=2)
    try:
        obs, _ = env.reset(seed=[10, 20])
        converted = prepare_canonical_observation(obs)
        assert converted["observation.image"].shape == (2, 3, 8, 8)
        assert converted["observation.image"].max() <= 1
        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 2), np.float32))
        assert obs["observation.state"].shape == (2, 2)
        assert env.call("task_description") == ("Move to the target",) * 2
        assert env.call("_max_episode_steps") == (5, 5)
    finally:
        env.close()


def test_robot_hold_does_not_advance_environment(server, tmp_path):
    from lerobot.robots.sim import SimRobot, SimRobotConfig
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    _, endpoint = server
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=endpoint), calibration_dir=tmp_path))
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_hold()
    try:
        robot.connect()
        assert len(wrapper.get_observation()) == 3
        wrapper.send_action({"command_0": 0.5, "command_1": 0})
        step = robot.latest.step[0]
        wrapper.hold()
        assert robot.latest.step[0] == step
    finally:
        robot.disconnect()


def test_sim_robot_registration(server, tmp_path):
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.sim import SimRobot, SimRobotConfig

    robot = make_robot_from_config(
        SimRobotConfig(env=ToyEnvConfig(endpoint=server[1]), calibration_dir=tmp_path)
    )
    try:
        assert isinstance(robot, SimRobot)
        robot.connect()
        action = dict.fromkeys(robot.action_features, 0.0)
        assert robot.send_action(action) == action
        assert robot.latest.step[0] == 1
    finally:
        robot.disconnect()


def test_libero_numpy_adapter_parity(monkeypatch):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from tests.utils import import_without_simulator

    canonical_observation = import_without_simulator("libero", monkeypatch).canonical_observation

    native = {
        "pixels": {"camera1": np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)},
        "robot_state": {
            "eef": {"pos": np.array([1, 2, 3], np.float32), "quat": np.array([0, 0, 0.6, 0.8], np.float32)},
            "gripper": {"qpos": np.array([0.1, 0.2], np.float32)},
        },
    }

    def batch(value):
        return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else value[None]

    expected = LiberoProcessorStep().observation(preprocess_observation(batch(native)))
    actual = canonical_observation(native)
    np.testing.assert_allclose(
        actual["observation.state"], expected["observation.state"][0].numpy(), atol=1e-6
    )
    np.testing.assert_array_equal(
        actual["observation.images.camera1"],
        (expected["observation.images.camera1"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
    )


def test_server_loss_fails_session(server):
    running, endpoint = server
    client = EnvClient(endpoint, timeout_s=0.1)
    try:
        client.open(1)
        running.stop.set()
        assert running.stop.is_set()
        time.sleep(0.03)
        with pytest.raises((TimeoutError, ProtocolError)):
            client.request("step", actions=np.zeros((1, 2), np.float32))
        assert client.failed
    finally:
        client.close()


def test_client_loss_releases_ownership(server):
    running, endpoint = server
    client = EnvClient(endpoint)
    client.open(1)
    time.sleep(0.03)
    client.transport.close()  # No close query: exercise liveliness cleanup.
    deadline = time.monotonic() + 2
    while running.sessions and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not running.sessions
    other = EnvClient(endpoint)
    try:
        other.open(1)
    finally:
        other.close()


def test_sim_robot_actions_follow_controller_order(server, tmp_path):
    from lerobot.robots.sim import SimRobot, SimRobotConfig

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=server[1]), calibration_dir=tmp_path))
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


def test_checkpoint_admission(server):
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    _, endpoint = server
    client = EnvClient(endpoint)
    try:
        descriptor = client.describe()
    finally:
        client.close()
    config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    descriptor.validate_checkpoint(config, {})
    # Acquisition resolution is independent of policy-side image preparation.
    for shape in ((3, 256, 256), (3, 480, 640)):
        config.input_features["observation.image"] = PolicyFeature(FeatureType.VISUAL, shape)
        descriptor.validate_checkpoint(config, {})
    for shape in ((4, 256, 256), (256, 256)):
        config.input_features["observation.image"] = PolicyFeature(FeatureType.VISUAL, shape)
        with pytest.raises(ValueError, match="shape mismatch"):
            descriptor.validate_checkpoint(config, {})
    # A checkpoint camera named differently is admitted through --rename_map, as for any robot.
    del config.input_features["observation.image"]
    config.input_features["observation.images.front"] = PolicyFeature(FeatureType.VISUAL, (3, 8, 8))
    with pytest.raises(ValueError, match="missing from the simulator"):
        descriptor.validate_checkpoint(config, {})
    descriptor.validate_checkpoint(config, {"observation.image": "observation.images.front"})
    # Checkpoint placeholder cameras are policy-owned: admission skips them whatever their shape.
    config.input_features["observation.images.empty_camera_0"] = PolicyFeature(
        FeatureType.VISUAL, (3, 480, 640)
    )
    descriptor.validate_checkpoint(config, {"observation.image": "observation.images.front"})
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (3,))
    with pytest.raises(ValueError, match="shape mismatch"):
        descriptor.validate_checkpoint(config, {"observation.image": "observation.images.front"})
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (2,))
    config.output_features["action"] = PolicyFeature(FeatureType.ACTION, (7,))
    with pytest.raises(ValueError, match="action dimension"):
        descriptor.validate_checkpoint(config, {"observation.image": "observation.images.front"})


def test_eval_loop_accepts_canonical_observations(server):
    from types import SimpleNamespace

    import torch

    from lerobot.env_server.client import BatchedEnv
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.scripts.lerobot_eval import rollout

    class Policy(torch.nn.Module):
        config = SimpleNamespace(use_amp=False)

        def reset(self):
            pass

        def select_action(self, observation):
            assert observation["observation.state"].shape == (2, 2)
            assert observation["observation.image"].shape == (2, 3, 8, 8)
            return torch.ones((2, 2), dtype=torch.float32) * 0.5

    _, endpoint = server
    env = BatchedEnv(EnvClient(endpoint), num_envs=2)
    identity = PolicyProcessorPipeline(steps=[])
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    action_identity = PolicyProcessorPipeline(
        steps=[], to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )
    try:
        result = rollout(
            env,
            Policy(),
            identity,
            identity,
            identity,
            action_identity,
            seeds=[10, 20],
            return_observations=True,
        )
        assert result["success"].any()
        assert result["observation"]["observation.image"].shape[2:] == (3, 8, 8)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["sync", "rtc"])
def test_sim_robot_local_inference_engines(server, tmp_path, mode):
    import torch

    from lerobot.inference import RTCInferenceEngine
    from lerobot.inference.sync import SyncInferenceEngine
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.robots.sim import SimRobot, SimRobotConfig
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.inference.rtc_helpers import ControlledPolicy, Pipeline
    from tests.utils import wait_for

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=server[1]), calibration_dir=tmp_path))
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


def test_sim_robot_remote_inference_engine(server, tmp_path):
    import torch

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.inference import PolicyRunner, RemoteInferenceConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.remote_inference.client import RemoteClient
    from lerobot.remote_inference.engine import RemoteInferenceEngine
    from lerobot.remote_inference.server import PolicyServer, SessionWorker
    from lerobot.robots.sim import SimRobot, SimRobotConfig
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.transport.zenoh import ZenohTransport
    from tests.inference.test_policy_runner import ConformingPolicy, tiny_config
    from tests.utils import wait_for

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=server[1]), calibration_dir=tmp_path))
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


def test_first_vector_reset_does_not_skip_initial_state(server, monkeypatch):
    from lerobot.env_server.client import BatchedEnv
    from tests.env_server.toy_env import ToyEnv

    original = ToyEnv.reset

    def reset(self, seed=None):
        self.reset_count = getattr(self, "reset_count", 0) + 1
        return original(self, seed=seed)

    monkeypatch.setattr(ToyEnv, "reset", reset)
    env = BatchedEnv(EnvClient(server[1]))
    # eval_policy_all prefetches the next task through `_ensure`; a prefetched open
    # would add an unseeded reset before the rollout's seeded one.
    assert not hasattr(env, "_ensure")
    try:
        env.reset(seed=123)
        served = server[0].sessions[env.client.session].env
        assert served._envs[0].reset_count == 1
        env.reset(seed=123)
        assert served._envs[0].reset_count == 2
    finally:
        env.close()


def test_rollout_rejects_checkpoint_schema_before_opening_environment(server, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.robots.sim import SimRobot, SimRobotConfig
    from lerobot.rollout import context

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=server[1]), calibration_dir=tmp_path))
    monkeypatch.setattr(context, "make_robot_from_config", lambda _: robot)
    cfg = SimpleNamespace(
        robot=robot.config,
        teleop=None,
        rename_map={},
        fps=robot.descriptor.fps,
        policy=SimpleNamespace(
            input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        ),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        context._connect_rollout_hardware(cfg)
    # A control rate other than the simulator's would replay actions too fast or too slow.
    cfg.fps = robot.descriptor.fps * 2
    with pytest.raises(ValueError, match="control rate"):
        context._connect_rollout_hardware(cfg)
    assert not server[0].sessions
    assert not robot.is_connected


def test_sim_robot_records_scalar_state_actions_and_rgb(server, tmp_path):
    from types import SimpleNamespace

    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.robots.sim import SimRobot, SimRobotConfig
    from lerobot.rollout import context
    from lerobot.rollout.configs import EpisodicStrategyConfig
    from lerobot.utils.feature_utils import build_dataset_frame

    robot = SimRobot(
        SimRobotConfig(env=ToyEnvConfig(endpoint=server[1]), calibration_dir=tmp_path / "calibration")
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


_ENVHUB_ENV = """
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Reach(gym.Env):
    observation_space = spaces.Dict(
        {
            "pixels": spaces.Dict({"front": spaces.Box(0, 255, (8, 8, 3), np.uint8)}),
            "agent_pos": spaces.Box(-10, 10, (2,), np.float32),
        }
    )
    action_space = spaces.Box(-1, 1, (2,), np.float32)
    task_description = "reach the goal"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-1, 1, 2).astype(np.float32)
        return self._obs(), {}

    def step(self, action):
        self.pos += action
        done = bool(np.linalg.norm(self.pos) > 3)
        return self._obs(), float(done), done, False, {"is_success": done}

    def _obs(self):
        return {"pixels": {"front": np.zeros((8, 8, 3), np.uint8)}, "agent_pos": self.pos.copy()}


def make_env(n_envs=1, use_async_envs=False):
    return {"reach_suite": {0: gym.vector.SyncVectorEnv([Reach] * n_envs)}}
"""


def test_envhub_serves_a_hub_repository(tmp_path, monkeypatch):
    from lerobot.sims.integrations import envhub
    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    env_file = tmp_path / "env.py"
    env_file.write_text(_ENVHUB_ENV)
    # Only the network download is stubbed; the consent check and URL parsing run for real.
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": "user/reach", "trust_remote_code": True},
    )
    env = VectorEnv(config, num_envs=2)
    try:
        assert env.tasks == {"reach_suite": ["0"]}
        assert {f.name: (f.shape, f.kind) for f in env.descriptor.features} == {
            "observation.images.front": ((8, 8, 3), "rgb"),
            "observation.state": ((2,), "tensor"),
        }
        assert env.descriptor.action_feature.shape == (2,)
        first = env.reset(seed=[3, 4])[0]["observation.state"]
        np.testing.assert_array_equal(env.reset(seed=[3, 4])[0]["observation.state"], first)
        info = env.step(np.full((2, 2), 0.5, np.float32))[4]
        assert info["task"] == ("reach the goal", "reach the goal")
    finally:
        env.close()
    # Running remote code needs explicit consent.
    config.kwargs["trust_remote_code"] = False
    with pytest.raises(RuntimeError, match="Refusing"):
        VectorEnv(config, num_envs=1)


def test_envhub_make_env_takes_the_options_of_the_command(tmp_path, monkeypatch):
    from lerobot.envs.configs import HubEnvConfig
    from lerobot.sims.integrations import envhub
    from lerobot.sims.vector_env import VectorEnv

    # The repository's make_env sizes its camera from an option of its own.
    env_file = tmp_path / "env.py"
    env_file.write_text(
        _ENVHUB_ENV
        + """
class SizedReach(Reach):
    def __init__(self, size):
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"front": spaces.Box(0, 255, (size, size, 3), np.uint8)}),
                "agent_pos": spaces.Box(-10, 10, (2,), np.float32),
            }
        )


def make_env(n_envs=1, use_async_envs=False, size=8):
    return {"reach_suite": {0: gym.vector.SyncVectorEnv([lambda: SizedReach(size)] * n_envs)}}
"""
    )
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = HubEnvConfig(hub_path="user/reach", trust_remote_code=True, make_env_kwargs={"size": 4})
    env = VectorEnv(config.sim_config())
    try:
        shapes = {f.name: f.shape for f in env.descriptor.features}
        assert shapes["observation.images.front"] == (4, 4, 3)
    finally:
        env.close()


def test_envhub_serves_a_local_directory(tmp_path, monkeypatch):
    from lerobot.sims.integrations import envhub
    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    (tmp_path / "env.py").write_text(_ENVHUB_ENV)

    def no_download(**kwargs):
        raise AssertionError("A local directory is not downloaded")

    monkeypatch.setattr(envhub, "hf_hub_download", no_download)
    env = VectorEnv(
        SimConfig(
            type="envhub",
            task="",
            semantics="envhub-v1",
            kwargs={"hub_path": str(tmp_path), "trust_remote_code": True},
        )
    )
    try:
        assert env.tasks == {"reach_suite": ["0"]}
    finally:
        env.close()


def test_envhub_module_runs_once_per_server_process(tmp_path, monkeypatch):
    import builtins

    from lerobot.sims.integrations import envhub
    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    # A simulator app that can start only once per process (e.g. Isaac Sim) lives in module state.
    env_file = tmp_path / "env.py"
    env_file.write_text("import builtins\nbuiltins.envhub_imports += 1\n" + _ENVHUB_ENV)
    monkeypatch.setattr(builtins, "envhub_imports", 0, raising=False)
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": "user/reach", "trust_remote_code": True},
    )
    # The probe and two sessions each build their environments.
    for _ in range(3):
        env = VectorEnv(config)
        env.reset(seed=[0])
        env.close()
    assert builtins.envhub_imports == 1


def test_envhub_needs_dict_observations(tmp_path, monkeypatch):
    from lerobot.sims.integrations import envhub
    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    env_file = tmp_path / "env.py"
    # A flat array observation, which local LeRobot execution does not accept either.
    env_file.write_text(
        "import gymnasium as gym\n"
        "def make_env(n_envs=1, use_async_envs=False):\n"
        "    return gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * n_envs)\n"
    )
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": "user/reach", "trust_remote_code": True},
    )
    with pytest.raises(ValueError, match="dict observations"):
        VectorEnv(config)


def test_envhub_settings_come_from_the_repository(tmp_path, monkeypatch):
    from lerobot.sims.integrations import envhub
    from lerobot.sims.types import SimConfig

    declared = (
        'LEROBOT_SIM = {"fps": 15, "control": "delta", "semantics": "reach-delta-v1",'
        ' "state_names": ["x", "y"], "action_names": ["dx", "dy"], "robot_type": "so_follower"}\n'
    )
    env_file = tmp_path / "env.py"
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    sim = {"type": "envhub", "task": "", "kwargs": {"hub_path": "user/reach", "trust_remote_code": True}}
    # Repository values fill the settings left out; an explicit value wins over them.
    env_file.write_text(_ENVHUB_ENV + declared)
    resolved = SimConfig.resolve({**sim, "fps": 50})
    assert (resolved.fps, resolved.control, resolved.semantics) == (50, "delta", "reach-delta-v1")
    assert (resolved.state_names, resolved.action_names) == (("x", "y"), ("dx", "dy"))
    assert resolved.robot_type == "so_follower"
    # Repositories without LEROBOT_SIM keep working with the defaults.
    env_file.write_text(_ENVHUB_ENV)
    resolved = SimConfig.resolve(sim)
    assert (resolved.fps, resolved.control, resolved.semantics) == (30, "position", "envhub-v1")
    # A typo in LEROBOT_SIM is an error, not a silently ignored setting.
    env_file.write_text(_ENVHUB_ENV + 'LEROBOT_SIM = {"fsp": 30}\n')
    with pytest.raises(ValueError, match="unknown keys"):
        SimConfig.resolve(sim)
