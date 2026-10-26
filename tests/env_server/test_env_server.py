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

"""Served environments: protocol, sessions, and parity with local execution."""

import subprocess
import sys
import time

import numpy as np
import pytest

from lerobot.env_server.client import EnvClient
from lerobot.transport.wire.protocol import ErrorCode, ProtocolError
from tests.sims.toy_env import ToyEnvConfig


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


def test_first_vector_reset_does_not_skip_initial_state(server, monkeypatch):
    from lerobot.env_server.client import BatchedEnv
    from tests.sims.toy_env import ToyEnv

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


def _episode(config: ToyEnvConfig) -> list[tuple]:
    env = config.create_env("reach", 0, num_envs=2)
    try:
        obs, info = env.reset(seed=[10, 20])
        results = [(obs, np.zeros(2, np.float32), np.zeros(2, bool), np.zeros(2, bool), info)]
        for actions in ([[2, 0], [0, 0]], [[0.1, 0.1], [0.1, 0.1]], [[0.5, 0], [0.5, 0]]):
            results.append(env.step(np.array(actions, np.float32)))
        return results
    finally:
        env.close()


def test_local_environment_matches_the_server(server):
    local = _episode(ToyEnvConfig())
    served = _episode(ToyEnvConfig(endpoint=server[1]))
    for expected, actual in zip(local, served, strict=True):
        for expected_array, actual_array in zip(expected[1:4], actual[1:4], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
        for name, value in expected[0].items():
            np.testing.assert_array_equal(actual[0][name], value)
        for key in ("is_success", "step"):
            np.testing.assert_array_equal(actual[4][key], expected[4][key])
        assert tuple(actual[4]["task"]) == tuple(expected[4]["task"])


def test_served_env_must_be_the_env_the_command_describes(server):
    config = ToyEnvConfig(endpoint=server[1], fps=30)  # the server's toy runs at 20 fps
    with pytest.raises(ValueError, match="fps: server 20"):
        config.create_envs(n_envs=1)
