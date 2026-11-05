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
import threading
import time

import numpy as np
import pytest

from lerobot.env_server import EnvClient, EnvServer, EnvServerConfig, RemoteVectorEnv
from lerobot.env_server.protocol import decode_descriptor, encode_descriptor, instance_prefix
from lerobot.envs import EnvClientConfig
from lerobot.sim import SimSettings
from lerobot.transport.protocol import ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import ZenohConfig
from tests.fixtures.sim_server import free_port
from tests.fixtures.toy_env import ToyEnv, ToyEnvConfig


def test_torch_free_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
import numpy as np
import lerobot.env_server.server, lerobot.sim.local_vector_env, lerobot.sim.integrations.envhub
import lerobot.scripts.lerobot_env_server
from lerobot.transport.protocol.client import query_reply, decode_reply
from lerobot.utils.feature_spec import FeatureSpec, validate_array
from lerobot.transport.protocol.envelope import Envelope, MessageType, correlate_reply, raise_peer_error, validate_reply
feature = FeatureSpec('state', (2,), 'float32', semantics='state-v1')
validate_array(np.zeros(2, np.float32), feature)
request = Envelope(MessageType.DESCRIBE, request_id='request')
reply = request.reply(MessageType.DESCRIPTOR, {})
validate_reply(reply, request, MessageType.DESCRIPTOR)
for prefix in ('torch', 'lerobot.policies', 'lerobot.remote_inference'):
    assert not any(
        module is not None and (name == prefix or name.startswith(prefix + '.'))
        for name, module in sys.modules.items()
    )
""",
        ],
        check=True,
    )


def test_a_client_ignores_descriptor_fields_it_does_not_know():
    # A newer server may describe more than this client knows.
    env = ToyEnvConfig().create_env()
    env.close()
    wire = {**encode_descriptor(env.descriptor), "joint_units": ["rad", "rad"]}
    assert decode_descriptor(wire) == env.descriptor


@pytest.mark.parametrize("use_async_envs", [False, True])
def test_batch_terminal_freeze_and_seed_parity(sim_server, use_async_envs):
    _, endpoint = sim_server
    client = EnvClient(endpoint)
    try:
        descriptor = client.describe()
        assert descriptor.action_feature.shape == (2,)
        client.open(2, use_async_envs=use_async_envs)
        first = client.request("reset", seeds=[10, 20])
        second = client.request("reset", seeds=[10, 20])
        np.testing.assert_array_equal(first.obs["observation.state"], second.obs["observation.state"])
        ended = client.request("step", actions=np.array([[2, 0], [0, 0]], np.float32))
        assert ended.is_success.tolist() == [True, False]
        assert ended.reward.tolist() == [1.0, 0.0]
        frozen = ended.obs["observation.state"][0].copy()
        for _ in range(5):
            result = client.request("step", actions=np.ones((2, 2), np.float32) * 0.1)
        np.testing.assert_array_equal(result.obs["observation.state"][0], frozen)
        assert result.reward[0] == 0.0  # The terminal reward is earned once, as with the old freeze wrapper.
        assert result.elapsed_steps.tolist() == [1, 5]
        assert result.truncated.tolist() == [False, True]
        frames = result.frames()
        assert len(frames) == 2 and frames[0].shape == (8, 8, 3)
    finally:
        client.close()


def test_admission_and_stale_generation(sim_server):
    _, endpoint = sim_server
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


def test_a_request_of_the_wrong_type_for_its_key_is_rejected(sim_server):
    client = EnvClient(sim_server[1])
    try:
        client.describe()
        open_key = instance_prefix(client.deployment, client.instance_id) + "/open"
        with pytest.raises(ProtocolError, match="takes open requests, not describe") as error:
            client._query(open_key, MessageType.DESCRIBE, {}, 5, MessageType.ACCEPTED)
        assert error.value.code == ErrorCode.MALFORMED
    finally:
        client.close()


def test_malformed_actions_and_seeds_are_rejected_without_failing_the_simulator(sim_server):
    client = EnvClient(sim_server[1])
    try:
        client.open(1)
        with pytest.raises(ProtocolError, match=r"got \(1, 3\), float32") as error:
            client.request("step", actions=np.zeros((1, 3), np.float32))
        assert error.value.code == ErrorCode.INCOMPATIBLE
        # The client stops after any error; the session on the server still steps.
        client.failed = False
        assert client.request("step", actions=np.zeros((1, 2), np.float32)).elapsed_steps[0] == 1
        # Bad seeds are rejected the same way, before the simulator resets.
        with pytest.raises(ProtocolError, match="one seed per env") as error:
            client.request("reset", seeds=[1, 2])
        assert error.value.code == ErrorCode.INCOMPATIBLE
        client.failed = False
        assert client.request("step", actions=np.zeros((1, 2), np.float32)).elapsed_steps[0] == 2
    finally:
        client.close()


def test_jpeg_observations_match_raw(sim_server):
    _, endpoint = sim_server
    obs = {}
    for encoding in ("raw", "jpeg"):
        env = RemoteVectorEnv(EnvClient(endpoint, encoding=encoding, jpeg_quality=95), num_envs=2)
        try:
            obs[encoding], _ = env.reset(seed=[10, 20])
        finally:
            env.close()
    raw, jpeg = obs["raw"]["observation.image"], obs["jpeg"]["observation.image"]
    assert jpeg.shape == raw.shape == (2, 8, 8, 3) and jpeg.dtype == np.uint8
    assert np.abs(jpeg.astype(int) - raw.astype(int)).mean() < 8
    np.testing.assert_array_equal(obs["jpeg"]["observation.state"], obs["raw"]["observation.state"])
    with pytest.raises(ValueError, match="raw or jpeg"):
        EnvClientConfig(endpoint, encoding="png")
    # The server checks the encoding too, since it comes over the network.
    client = EnvClient(endpoint, encoding="png")
    with pytest.raises(ProtocolError, match="raw or jpeg"):
        client.open(1)
    client.close()


def test_the_client_payload_limit_bounds_the_batch(sim_server):
    _, endpoint = sim_server
    # A toy env observes 200 bytes, and the server keeps 64 KiB of each reply for the rest: two envs fit, three don't.
    limit = 64 * 1024 + 500
    for num_envs, fits in ((2, True), (3, False)):
        client = EnvClient(endpoint, max_payload_bytes=limit)
        try:
            if fits:
                assert client.open(num_envs).obs["observation.state"].shape == (num_envs, 2)
            else:
                with pytest.raises(ProtocolError, match="exceeds the payload budget") as error:
                    client.open(num_envs)
                assert error.value.code == ErrorCode.INCOMPATIBLE
        finally:
            client.close()


def test_server_loss_fails_session(sim_server):
    running, endpoint = sim_server
    client = EnvClient(endpoint, timeout_s=0.1)
    try:
        client.open(1)
        running.stop()
        time.sleep(0.03)
        with pytest.raises((TimeoutError, ProtocolError)):
            client.request("step", actions=np.zeros((1, 2), np.float32))
        assert client.failed
    finally:
        client.close()


def test_closing_after_the_server_is_gone_neither_waits_nor_raises(sim_server):
    running, endpoint = sim_server
    client = EnvClient(endpoint, timeout_s=60, handshake_timeout_s=0.5)
    client.open(1)
    running.stop()
    time.sleep(0.05)
    start = time.monotonic()
    client.close()
    assert time.monotonic() - start < 5 and not client.session_id


def test_client_loss_releases_ownership(sim_server):
    running, endpoint = sim_server
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


def test_eval_loop_accepts_canonical_observations(sim_server):
    from types import SimpleNamespace

    import torch

    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.scripts.lerobot_eval import rollout

    class Policy(torch.nn.Module):
        config = SimpleNamespace(use_amp=False)

        def reset(self):
            pass

        def select_action(self, observation):
            assert observation["observation.state"].shape == (2, 2)
            assert observation["observation.image"].shape == (2, 3, 8, 8)
            return torch.ones((2, 2), dtype=torch.float32) * 0.5

    env = RemoteVectorEnv(EnvClient(sim_server[1]), num_envs=2)
    identity = PolicyProcessorPipeline(steps=[])
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


def test_a_session_whose_first_reset_outlasts_the_presence_timeout_stays_open(sim_server, monkeypatch):
    # The client can only announce itself once the open reply, which carries the first reset, arrives.
    running, endpoint = sim_server
    timeout = running.config.initial_presence_timeout_s
    original = ToyEnv.reset

    def slow_reset(self, seed=None):
        time.sleep(timeout * 1.5)
        return original(self, seed=seed)

    monkeypatch.setattr(ToyEnv, "reset", slow_reset)
    client = EnvClient(endpoint)
    try:
        client.open(1)
        # A policy's first inference can take longer than the presence timeout.
        time.sleep(timeout * 1.5)
        client.request("step", actions=np.zeros((1, 2), np.float32))
    finally:
        client.close()


def test_first_vector_reset_does_not_skip_initial_state(sim_server, monkeypatch):
    # The first reset opens the session, which resets the simulator once; it must not reset it twice.
    resets = []
    original = ToyEnv.reset

    def reset(self, seed=None):
        resets.append(seed)
        return original(self, seed=seed)

    monkeypatch.setattr(ToyEnv, "reset", reset)
    env = RemoteVectorEnv(EnvClient(sim_server[1]))
    try:
        env.reset(seed=123)
        assert resets == [123]
        env.reset(seed=123)
        assert resets == [123, 123]
    finally:
        env.close()


def _episode(config: ToyEnvConfig) -> list[tuple]:
    env = config.create_env("reach", 0, n_envs=2)
    try:
        obs, info = env.reset(seed=[10, 20])
        results = [(obs, np.zeros(2, np.float32), np.zeros(2, bool), np.zeros(2, bool), info)]
        for actions in ([[2, 0], [0, 0]], [[0.1, 0.1], [0.1, 0.1]], [[0.5, 0], [0.5, 0]]):
            results.append(env.step(np.array(actions, np.float32)))
        return results
    finally:
        env.close()


def test_local_environment_matches_the_server(sim_server):
    local = _episode(ToyEnvConfig())
    served = _episode(ToyEnvConfig(server=EnvClientConfig(sim_server[1])))
    for expected, actual in zip(local, served, strict=True):
        for expected_array, actual_array in zip(expected[1:4], actual[1:4], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
        for name, value in expected[0].items():
            np.testing.assert_array_equal(actual[0][name], value)
        for key in ("is_success", "elapsed_steps"):
            np.testing.assert_array_equal(actual[4][key], expected[4][key])
        assert tuple(actual[4]["task"]) == tuple(expected[4]["task"])
        assert actual[4]["metrics"].keys() == expected[4]["metrics"].keys()
        for name, values in expected[4]["metrics"].items():
            np.testing.assert_array_equal(actual[4]["metrics"][name], values)


def test_served_env_must_be_the_env_the_command_describes(sim_server):
    config = ToyEnvConfig(server=EnvClientConfig(sim_server[1]), fps=30)  # the server's toy runs at 20 fps
    with pytest.raises(ValueError, match="fps: server 20"):
        config.create_envs(n_envs=1)


@pytest.mark.parametrize("task_id", [None, 0])
def test_the_first_session_reuses_the_environment_built_to_describe(sim_server, task_id):
    # Without a task ID a session runs the suite's first task, which the startup env already runs.
    before = ToyEnv.builds
    client = EnvClient(sim_server[1])
    try:
        client.open(1, task_id=task_id)
        client.request("step", actions=np.zeros((1, 2), np.float32))
        assert ToyEnv.builds == before
    finally:
        client.close()


def test_a_bad_first_result_fails_the_session(sim_server, monkeypatch):
    def reject(*args):
        raise ValueError("bad first result")

    monkeypatch.setattr("lerobot.env_server.client.validate_result", reject)
    client = EnvClient(sim_server[1])
    try:
        with pytest.raises(ValueError, match="bad first result"):
            client.open(1)
        assert client.failed
    finally:
        client.close()


def test_a_wrong_deployment_names_the_one_served(sim_server):
    config = ToyEnvConfig(server=EnvClientConfig(sim_server[1], deployment="other", handshake_timeout_s=0.5))
    with pytest.raises(ProtocolError, match="serves deployment 'toy', not 'other'") as error:
        config.create_envs(n_envs=1)
    assert "--env.server.deployment=toy" in str(error.value)
    assert "start a server for 'other'" in str(error.value)


def test_a_server_that_finishes_starting_during_the_handshake_is_described(monkeypatch):
    # A slow simulator (e.g. Isaac Sim) can come up after the client's first describe query has timed out.
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    original = ToyEnv.__init__

    def slow_init(self):
        time.sleep(1.0)
        original(self)

    monkeypatch.setattr(ToyEnv, "__init__", slow_init)
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    server = EnvServer(
        EnvServerConfig(
            sim=SimSettings(**ToyEnvConfig().sim_settings()),
            deployment="toy",
            zenoh=ZenohConfig(listen_endpoints=[endpoint]),
        )
    )
    thread = threading.Thread(target=server.serve)
    thread.start()
    client = EnvClient(endpoint, "toy", handshake_timeout_s=0.7)
    try:
        assert client.describe().sim_type == "toy"
    finally:
        client.close()
        server.stop()
        thread.join(5)
    assert not thread.is_alive()


def test_an_absent_server_is_reported_within_the_handshake_timeout():
    pytest.importorskip("zenoh")
    client = EnvClient(f"tcp/127.0.0.1:{free_port()}", "toy", handshake_timeout_s=0.5)
    start = time.monotonic()
    try:
        with pytest.raises(ConnectionError, match="Could not connect to an env server"):
            client.describe()
        assert time.monotonic() - start < 5
    finally:
        client.close()
